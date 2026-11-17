#pragma once

// The append-only increment that sits beside a session file: `<name>.conv` is the base image a conversation was
// saved at, `<name>.conv.delta` holds the K/V bytes of what came after it.  A conversation that grows by one turn a
// minute then costs that turn's bytes instead of the whole image (18.2 KiB a token at this geometry), and the reader
// sees exactly the image the base would have had if it had been rewritten.
//
// Format (little-endian, fixed width, header-only so the disk tier can include it without a new translation unit):
//
//   header:  magic[18] "STRATA-CONV-DELTA\0", u32 version, u64 model, u64 config, u64 base_fp, i64 base_tokens
//   segment: u32 seg_magic "DELT", u32 seg_version, i64 cell_from, i64 cell_to, i64 live_tokens,
//            i64 prev_live_tokens, u64 payload_bytes, <payload>, u64 seg_hash
//     payload: u64 id_count, i32 ids[id_count]                              ; tokens [prev_live, live)
//              u64 state_len[5], bytes[state_len]                          ; gdn, ple, tails, dead, block_pos
//              u64 kv_layers, per layer: i64 x7 (format..idx_dim), u8 part_mask,
//                  per part: u64 offset_in_part, u64 len, bytes[len]
//              u64 checkpoint_count, per checkpoint: the session file's own checkpoint encoding
//
// Two invariants make it safe:
//   * a segment only ever appends: an existing byte is never moved, so a torn tail costs the tail, not the file;
//   * K/V part sizes are linear in the padded cell count (conversation_snapshot.cpp layout()), so `cell_from`
//     page-aligns the start and `offset = cells_before * (part_size / cells)` is exact.  The base's last partial
//     page is rewritten by the segment because its unused cells belong to no saved state.

#include "strata/core/conversation_file.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace strata::core {

inline constexpr char kDeltaMagic[18] = {'S', 'T', 'R', 'A', 'T', 'A', '-', 'C', 'O', 'N',
                                         'V', '-', 'D', 'E', 'L', 'T', 'A', '\0'};
inline constexpr uint32_t kDeltaVersion = 1;
inline constexpr uint32_t kDeltaSegmentMagic = 0x44454C54u;   // "DELT"
// v2: the segment header carries `idx_block` and `row_bytes` after `payload_bytes`, so the pooled indexer part is a
// row range of the growing conversation instead of the whole block (v1 stored it whole: ~4.5 MB a layer a segment).
// A v1 segment is not read: the reader stops there, the base image stands, and the next append rewrites the chain.
inline constexpr uint32_t kDeltaSegmentVersion = 2;
inline constexpr uint64_t kDeltaHeaderBytes = 18 + 4 + 8 + 8 + 8 + 8;   // 54
inline constexpr uint64_t kDeltaSegmentHeaderBytes = 4 + 4 + 8 + 8 + 8 + 8 + 8 + 8 + 8;   // 64

// What one segment adds, as the reader found it.
struct DeltaSegmentInfo {
    int64_t cell_from = 0, cell_to = 0, live_tokens = 0, prev_live_tokens = 0;
    uint64_t payload_bytes = 0;
};

// What a delta file may hold, and what a read may walk into.
struct SessionDeltaLimits {
    uint64_t max_file_bytes = UINT64_MAX;
    uint64_t max_segments = 65536;
    uint64_t max_payload_bytes = UINT64_MAX;
};

// The page start a segment rewrites from: the base's last page is partially filled, so it is written again.
inline int64_t delta_cell_from(int64_t live_tokens, int64_t page_size) {
    if (page_size <= 0 || live_tokens <= 0) return 0;
    return (live_tokens / page_size) * page_size;
}
inline int64_t delta_cell_to(int64_t live_tokens, int64_t page_size) {
    if (page_size <= 0 || live_tokens <= 0) return live_tokens;
    return ((live_tokens + page_size - 1) / page_size) * page_size;
}

inline std::string session_delta_path(const std::string& session_path) { return session_path + ".delta"; }

// Ties the increment to the base file it extends: a base rewritten (compacted) or replaced makes the increment
// obsolete rather than wrong.
inline uint64_t session_delta_base_fingerprint(const SessionFileIdentity& id, int64_t base_tokens, uint64_t base_bytes) {
    SessionIdentityBuilder b(0x44454c54414241ull);
    b.u64("model", id.model);
    b.u64("config", id.config);
    b.i64("tokens", base_tokens);
    b.u64("bytes", base_bytes);
    return b.digest();
}

namespace conversation_delta_detail {

inline bool seek_end(std::FILE* f) {
#ifdef _WIN32
    return _fseeki64(f, 0, SEEK_END) == 0;
#else
    return std::fseek(f, 0, SEEK_END) == 0;
#endif
}
inline bool seek_to(std::FILE* f, uint64_t at) {
#ifdef _WIN32
    return _fseeki64(f, (__int64) at, SEEK_SET) == 0;
#else
    return fseeko(f, (off_t) at, SEEK_SET) == 0;
#endif
}
inline uint64_t file_size_of(std::FILE* f) {
    if (!seek_end(f)) return 0;
#ifdef _WIN32
    const __int64 at = _ftelli64(f);
#else
    const off_t at = ftello(f);
#endif
    return at < 0 ? 0 : (uint64_t) at;
}
inline bool flush_durable(std::FILE* f) {
    if (std::fflush(f) != 0) return false;
#ifdef _WIN32
    return _commit(_fileno(f)) == 0;
#else
    return ::fsync(fileno(f)) == 0;
#endif
}
inline bool truncate_to(std::FILE* f, uint64_t size) {
    std::fflush(f);
#ifdef _WIN32
    return _chsize_s(_fileno(f), (__int64) size) == 0;
#else
    return ::ftruncate(fileno(f), (off_t) size) == 0;
#endif
}

// Writes with a running hash: everything a segment holds goes through here, so seg_hash covers it.
struct writer {
    std::FILE* f = nullptr;
    SessionHasher hash{0x64656c7461ull};
    bool ok = true;
    uint64_t written = 0;
    void raw(const void* p, size_t n) {
        if (!ok || n == 0) return;
        hash.update(p, n);
        if (std::fwrite(p, 1, n, f) != n) {
            ok = false;
            return;
        }
        written += n;
    }
    void u32(uint32_t v) { raw(&v, 4); }
    void u64(uint64_t v) { raw(&v, 8); }
    void i64(int64_t v) { raw(&v, 8); }
    void blob(const void* p, size_t n) {
        u64(n);
        raw(p, n);
    }
    void blob(const std::vector<uint8_t>& v) { blob(v.data(), v.size()); }
};

// Reads with a running hash, up to a stated bound: a payload byte past the segment's own length is a broken file.
struct reader {
    std::FILE* f = nullptr;
    uint64_t size = 0, at = 0;
    SessionHasher hash{0x64656c7461ull};
    bool ok = true;
    std::string error;
    bool fail(const char* m) {
        if (error.empty()) error = std::string("session delta: ") + m;
        ok = false;
        return false;
    }
    bool left(uint64_t n) const { return n <= size - at; }
    bool raw(void* p, size_t n) {
        if (!ok) return false;
        if (!left(n)) return fail("the file ends early");
        if (n != 0 && std::fread(p, 1, n, f) != n) return fail("read failed");
        hash.update(p, n);
        at += n;
        return true;
    }
    bool plain(void* p, size_t n) {
        if (!ok) return false;
        if (!left(n)) return fail("the file ends early");
        if (n != 0 && std::fread(p, 1, n, f) != n) return fail("read failed");
        at += n;
        return true;
    }
    bool u32(uint32_t& v) { return raw(&v, 4); }
    bool u64(uint64_t& v) { return raw(&v, 8); }
    bool i64(int64_t& v) { return raw(&v, 8); }
    bool skip(uint64_t n) {
        if (!ok) return false;
        if (!left(n)) return fail("the file ends early");
        at += n;
        return seek_to(f, at);
    }
};

struct header {
    int64_t base_tokens = 0;
    uint64_t base_fp = 0;
};

inline bool write_header(std::FILE* f, const SessionFileIdentity& id, const header& h) {
    writer w;
    w.f = f;
    w.raw(kDeltaMagic, sizeof kDeltaMagic);
    w.u32(kDeltaVersion);
    w.u64(id.model);
    w.u64(id.config);
    w.u64(h.base_fp);
    w.i64(h.base_tokens);
    return w.ok;
}

// Reads and checks the file header.  `missing` says the file is not there at all (no increment, not an error).
inline bool read_header(std::FILE* f, const SessionFileIdentity& id, header& h, bool& missing, std::string& error) {
    missing = false;
    char magic[sizeof kDeltaMagic] = {};
    if (std::fread(magic, 1, sizeof magic, f) != sizeof magic) {
        missing = true;
        return true;
    }
    if (std::memcmp(magic, kDeltaMagic, sizeof kDeltaMagic) != 0) {
        error = "session delta: not a delta file";
        return false;
    }
    uint32_t version = 0, seg = 0;
    uint64_t model = 0, config = 0, fp = 0;
    int64_t tokens = 0;
    if (std::fread(&version, 4, 1, f) != 1 || std::fread(&model, 8, 1, f) != 1 ||
        std::fread(&config, 8, 1, f) != 1 || std::fread(&fp, 8, 1, f) != 1 || std::fread(&tokens, 8, 1, f) != 1) {
        error = "session delta: the header ends early";
        return false;
    }
    (void) seg;
    if (version != kDeltaVersion) {
        error = "session delta: unknown version " + std::to_string(version);
        return false;
    }
    if (model != id.model || config != id.config) {
        error = "session delta: written with another model or configuration";
        return false;
    }
    h.base_tokens = tokens;
    h.base_fp = fp;
    return true;
}

}   // namespace conversation_delta_detail

// The segments a delta file holds, in order; stops at the first incomplete or foreign one (a torn append).  No
// payload is read: this is the cheap "is there anything newer, and up to where" question the disk tier asks first.
inline bool session_delta_read(const std::string& delta_path, const SessionFileIdentity& id, int64_t base_tokens,
                               uint64_t base_bytes, std::vector<DeltaSegmentInfo>& out, uint64_t& valid_bytes,
                               std::string& error, const SessionDeltaLimits& limits = {}) {
    namespace d = conversation_delta_detail;
    out.clear();
    valid_bytes = 0;
    std::FILE* f = std::fopen(delta_path.c_str(), "rb");
    if (!f) return true;   // no increment
    struct close_guard {
        std::FILE* f;
        ~close_guard() { std::fclose(f); }
    } guard{f};
    d::header h;
    bool missing = false;
    if (!d::read_header(f, id, h, missing, error)) return false;
    if (missing) {
        out.clear();
        return true;
    }
    const uint64_t bytes = d::file_size_of(f);
    if (bytes > limits.max_file_bytes) {
        error = "session delta: file is larger than this engine's limit";
        return false;
    }
    if (h.base_tokens != base_tokens || h.base_fp != session_delta_base_fingerprint(id, base_tokens, base_bytes)) {
        error = "session delta: it belongs to another base file";
        return false;
    }
    valid_bytes = kDeltaHeaderBytes;
    for (uint64_t n = 0; n < limits.max_segments; ++n) {
        if (!d::seek_to(f, valid_bytes)) break;
        DeltaSegmentInfo info;
        uint32_t magic = 0, version = 0;
        if (std::fread(&magic, 4, 1, f) != 1) break;
        if (magic != kDeltaSegmentMagic) break;   // torn tail or not our bytes
        if (std::fread(&version, 4, 1, f) != 1 || version != kDeltaSegmentVersion) break;
        if (std::fread(&info.cell_from, 8, 1, f) != 1 || std::fread(&info.cell_to, 8, 1, f) != 1 ||
            std::fread(&info.live_tokens, 8, 1, f) != 1 || std::fread(&info.prev_live_tokens, 8, 1, f) != 1 ||
            std::fread(&info.payload_bytes, 8, 1, f) != 1)
            break;
        int64_t seg_idx_block = 0, seg_row_bytes = 0;   // v2 header: the pooled row range this segment carries
        if (std::fread(&seg_idx_block, 8, 1, f) != 1 || std::fread(&seg_row_bytes, 8, 1, f) != 1) break;
        if (info.payload_bytes > limits.max_payload_bytes) break;
        const uint64_t end = valid_bytes + kDeltaSegmentHeaderBytes + info.payload_bytes + 8;
        if (end > bytes) break;   // the append did not finish
        out.push_back(info);
        valid_bytes = end;
    }
    return true;
}

// Appends one segment: the K/V bytes of [prev_live_tokens, meta.live.ids.size()) of the conversation `meta` describes,
// read straight from `kv` (device or pinned host pool) — no host copy of the image is made here.
inline bool session_delta_append(const std::string& delta_path, const SavedConversation& meta,
                                 const std::vector<SessionKvSource>& kv, const SessionFileIdentity& id,
                                 int64_t base_tokens, uint64_t base_bytes, int64_t prev_live_tokens, size_t& bytes,
                                 std::string& error, const SessionWriteOptions& options = {}) {
    namespace d = conversation_delta_detail;
    bytes = 0;
    if (!meta.kv.empty() || kv.empty()) {
        error = "session delta: the increment needs K/V sources and a K/V-free image";
        return false;
    }
    if (!meta.stage_images.empty()) {
        error = "session delta: a layer split saves no increment";
        return false;
    }
    const int64_t live = (int64_t) meta.live.ids.size();
    if (live <= prev_live_tokens) {
        error = "session delta: nothing new to append";
        return false;
    }
    const int64_t page = kv.front().page_size;
    const int64_t cell_from = delta_cell_from(prev_live_tokens, page);
    const int64_t cell_to = delta_cell_to(live, page);
    // The pooled indexer rows this segment rewrites: one row per `idx_block` tokens, so the row that holds the
    // first new token is written again (the base may have it half full) and the last one is the new end.
    const int64_t idx_block = kv.front().idx_block;
    const int64_t row_bytes = kv.front().idx_dim > 0 ? kv.front().idx_dim * 4 : 0;
    const int64_t row_from = idx_block > 0 && row_bytes > 0 ? prev_live_tokens / idx_block : 0;
    const int64_t row_to = idx_block > 0 && row_bytes > 0 ? live / idx_block + 1 : 0;

    // What each layer and part carries: computed once, used for the size field the reader skips by and for the write.
    // Part sizes are linear in the padded cell count, so `(size / cells) * cells_before + (size % cells) * cells_before
    // / cells` is exact and cannot overflow.  The pooled indexer part is linear in its rows too: one row of
    // `row_bytes` per `idx_block` tokens, so a segment writes only the rows it lands in and the new ones after them
    // (writing that block whole costs ~4.5 MB a layer a segment at this geometry, more than the K/V increment).
    struct part_plan {
        uint64_t offset = 0, len = 0;
    };
    std::vector<std::array<part_plan, 5>> plan(kv.size());
    for (size_t i = 0; i < kv.size(); ++i)
        for (size_t part = 0; part < 5; ++part) {
            const uint64_t size = kv[i].sizes[part];
            if (size == 0) continue;
            uint64_t offset = 0, len = size;
            if (part == 4 && row_bytes > 0 && row_to > row_from) {
                const uint64_t from = (uint64_t) row_from * (uint64_t) row_bytes;
                const uint64_t to = (uint64_t) row_to * (uint64_t) row_bytes;
                offset = from < size ? from : size;
                len = to > offset ? std::min(to, size) - offset : 0;
            } else if (part != 4 && kv[i].cells > 0) {
                // The byte offset of a cell, exactly (part sizes are linear in cells but not necessarily divisible):
                // the increment is the cells [cell_from, cell_to) only.  NOT "from offset to the end of the source":
                // a source's cells can be the session's allocated cells, far past this conversation's own end.
                const uint64_t cells = (uint64_t) kv[i].cells;
                auto at = [size, cells](uint64_t cell) {
                    return (size / cells) * cell + ((size % cells) * cell) / cells;
                };
                offset = at((uint64_t) cell_from);
                len = at((uint64_t) cell_to) - offset;
            }
            plan[i][part] = {offset, len};
        }
    if (std::getenv("STRATA_DELTA_TRACE") != nullptr && !kv.empty())
        std::fprintf(stderr,
                     "strata serve: conversation delta: cells %lld (source) vs [%lld, %lld), layer0 sizes %llu %llu %llu %llu %llu, write %llu %llu %llu %llu %llu\n",
                     (long long) kv.front().cells, (long long) cell_from, (long long) cell_to,
                     (unsigned long long) kv.front().sizes[0], (unsigned long long) kv.front().sizes[1],
                     (unsigned long long) kv.front().sizes[2], (unsigned long long) kv.front().sizes[3],
                     (unsigned long long) kv.front().sizes[4], (unsigned long long) plan.front()[0].len,
                     (unsigned long long) plan.front()[1].len, (unsigned long long) plan.front()[2].len,
                     (unsigned long long) plan.front()[3].len, (unsigned long long) plan.front()[4].len);

    // the payload size first: the segment header carries it, and a reader skips a segment with it
    uint64_t payload = 8 + (uint64_t) (live - prev_live_tokens) * 4;
    for (const auto* v : {&meta.live.gdn, &meta.live.ple, &meta.live.tails, &meta.live.dead, &meta.live.block_pos})
        payload += 8 + v->size();
    payload += 8;   // kv layer count
    for (size_t i = 0; i < kv.size(); ++i) {
        payload += 7 * 8 + 1;
        for (const auto& p : plan[i]) payload += 8 + 8 + p.len;
    }
    payload += 8;   // checkpoint count (a segment normally carries none)

    // open (or create) and validate against the base it extends
    std::FILE* f = std::fopen(delta_path.c_str(), "r+b");
    bool fresh = false;
    if (!f) {
        f = std::fopen(delta_path.c_str(), "w+b");
        if (!f) {
            error = "session delta: cannot create " + delta_path;
            return false;
        }
        fresh = true;
    }
    struct close_guard {
        std::FILE* f;
        ~close_guard() { std::fclose(f); }
    } guard{f};
    d::header h;
    if (fresh) {
        h.base_tokens = base_tokens;
        h.base_fp = session_delta_base_fingerprint(id, base_tokens, base_bytes);
        if (!d::write_header(f, id, h) || !d::flush_durable(f)) {
            error = "session delta: cannot write the header of " + delta_path;
            return false;
        }
    } else {
        bool missing = false;
        if (!d::read_header(f, id, h, missing, error)) return false;
        if (missing || h.base_tokens != base_tokens ||
            h.base_fp != session_delta_base_fingerprint(id, base_tokens, base_bytes)) {
            error = "session delta: it belongs to another base file";
            return false;
        }
    }
    // self-heal: a torn append is dropped before this one is added
    uint64_t valid = 0;
    std::vector<DeltaSegmentInfo> have;
    {
        std::string why;
        if (!session_delta_read(delta_path, id, base_tokens, base_bytes, have, valid, why)) {
            error = why;
            return false;
        }
        if (have.empty()) valid = kDeltaHeaderBytes;
        else if (!have.empty() && have.back().live_tokens != prev_live_tokens) {
            // the caller's previous point must be the file's newest: another writer advanced it
            error = "session delta: the increment has newer segments than this save assumes";
            return false;
        }
    }
    const uint64_t before = d::file_size_of(f);
    if (before != valid && !d::truncate_to(f, valid)) {
        error = "session delta: cannot drop a torn tail of " + delta_path;
        return false;
    }
    if (!d::seek_to(f, valid)) {
        error = "session delta: cannot seek in " + delta_path;
        return false;
    }
    d::writer w;
    w.f = f;
    w.u32(kDeltaSegmentMagic);
    w.u32(kDeltaSegmentVersion);
    w.i64(cell_from);
    w.i64(cell_to);
    w.i64(live);
    w.i64(prev_live_tokens);
    w.u64(payload);
    w.i64(idx_block);   // v2: the pooled rows [prev / idx_block, live / idx_block + 1) are what part 4 carries
    w.i64(row_bytes);
    w.u64((uint64_t) (live - prev_live_tokens));
    for (int64_t t = prev_live_tokens; t < live; ++t) w.raw(&meta.live.ids[(size_t) t], 4);
    w.blob(meta.live.gdn);
    w.blob(meta.live.ple);
    w.blob(meta.live.tails);
    w.blob(meta.live.dead);
    w.blob(meta.live.block_pos);
    w.u64(kv.size());
    std::vector<uint8_t> buffer(16u << 20);
    for (size_t i = 0; i < kv.size() && w.ok; ++i) {
        const SessionKvSource& s = kv[i];
        for (int64_t v : {(int64_t) s.format, s.cells, s.heads, s.head_dim, s.page_size, s.pooled_rows, s.idx_dim})
            w.i64(v);
        uint8_t mask = 0;
        for (size_t part = 0; part < 5; ++part)
            if (plan[i][part].len != 0) mask |= (uint8_t) (1u << part);
        w.raw(&mask, 1);
        for (size_t part = 0; part < 5 && w.ok; ++part) {
            w.u64(plan[i][part].offset);
            w.u64(plan[i][part].len);
            uint64_t left = plan[i][part].len;
            uint64_t at = plan[i][part].offset;
            while (left && w.ok) {
                const size_t chunk = (size_t) std::min<uint64_t>(buffer.size(), left);
                if (!s.read || !s.read(part, (size_t) at, buffer.data(), chunk)) {
                    w.ok = false;
                    error = "session delta: K/V source read" + (s.error ? (": " + s.error()) : std::string());
                    break;
                }
                w.raw(buffer.data(), chunk);
                at += chunk;
                left -= chunk;
            }
        }
    }
    w.u64(0);   // no checkpoint in a routine segment
    const uint64_t seg_hash = w.hash.digest();
    if (w.ok) {
        if (std::fwrite(&seg_hash, 1, 8, f) != 8) w.ok = false;
    }
    if (!w.ok || !d::flush_durable(f)) {
        if (error.empty()) error = "session delta: writing " + delta_path + " failed";
        d::truncate_to(f, valid);   // the base and the earlier segments stay readable
        return false;
    }
    bytes = (size_t) (d::file_size_of(f) - valid);
    (void) options;
    return true;
}

// Applies every segment that still fits the chain onto an image the caller already read from the base file.  The
// result is byte-for-byte what a full write of the same conversation would read back as.  `delta_used` says whether
// any segment was applied (false when the increment belongs elsewhere, is torn from its first segment, or was already
// covered by the base); `error` then says why it was left out — the base image stands either way.
inline bool session_delta_overlay(const std::string& base_path, const std::string& delta_path,
                                  const SessionFileIdentity& id, SavedConversation& image, uint64_t& delta_bytes,
                                  std::string& error, bool& delta_used,
                                  const SessionDeltaLimits& delta_limits = {}) {
    namespace d = conversation_delta_detail;
    delta_used = false;
    delta_bytes = 0;
    std::error_code ec;
    const uint64_t base_bytes = (uint64_t) std::filesystem::file_size(base_path, ec);
    if (ec) {
        error = "session delta: cannot size " + base_path;
        return true;   // the base image is in hand: use it
    }
    const int64_t base_tokens = (int64_t) image.live.ids.size();
    std::vector<DeltaSegmentInfo> segs;
    uint64_t valid = 0;
    std::string why;
    if (!session_delta_read(delta_path, id, base_tokens, base_bytes, segs, valid, why, delta_limits)) {
        error = why;   // an increment that cannot be read is dropped, the base stands
        return true;
    }
    if (segs.empty()) return true;

    std::FILE* f = std::fopen(delta_path.c_str(), "rb");
    if (!f) {
        error = "session delta: cannot open " + delta_path;
        return true;
    }
    struct close_guard {
        std::FILE* f;
        ~close_guard() { std::fclose(f); }
    } guard{f};
    const uint64_t file_bytes = d::file_size_of(f);
    uint64_t at = kDeltaHeaderBytes;
    for (const auto& seg : segs) {
        if (seg.live_tokens <= (int64_t) image.live.ids.size()) continue;   // already covered
        if (seg.prev_live_tokens != (int64_t) image.live.ids.size()) break;  // the chain does not meet the base
        if (!d::seek_to(f, at)) break;
        d::reader r;
        r.f = f;
        r.size = file_bytes;
        r.at = at;
        uint32_t magic = 0, version = 0;
        int64_t cell_from = 0, cell_to = 0, live = 0, prev = 0;
        uint64_t payload = 0;
        int64_t seg_idx_block = 0, seg_row_bytes = 0;
        if (!r.u32(magic) || magic != kDeltaSegmentMagic) break;
        if (!r.u32(version) || version != kDeltaSegmentVersion) break;
        if (!r.i64(cell_from) || !r.i64(cell_to) || !r.i64(live) || !r.i64(prev) || !r.u64(payload)) break;
        if (!r.i64(seg_idx_block) || !r.i64(seg_row_bytes)) break;
        if (image.kv.empty()) break;
        const int64_t page = image.kv.front().page_size;
        if (cell_to != delta_cell_to(live, page)) break;
        uint64_t ids_n = 0;
        if (!r.u64(ids_n)) break;
        // ids and state arrays: small, read into fresh vectors
        std::vector<int32_t> ids((size_t) ids_n);
        if (ids_n > 0 && !r.raw(ids.data(), ids.size() * 4)) break;
        std::vector<uint8_t>* state[5] = {&image.live.gdn, &image.live.ple, &image.live.tails, &image.live.dead,
                                          &image.live.block_pos};
        bool state_ok = true;
        for (size_t s = 0; s < 5 && state_ok; ++s) {
            uint64_t n = 0;
            if (!r.u64(n)) {
                state_ok = false;
                break;
            }
            std::vector<uint8_t> v((size_t) n);
            if (n > 0 && !r.raw(v.data(), v.size())) {
                state_ok = false;
                break;
            }
            *state[s] = std::move(v);
        }
        if (!state_ok) break;
        uint64_t layers = 0;
        if (!r.u64(layers) || layers != image.kv.size()) break;
        bool kv_ok = true;
        for (size_t i = 0; i < image.kv.size() && kv_ok; ++i) {
            ConversationKv& k = image.kv[i];
            int64_t f7[7] = {};
            for (auto& v : f7)
                if (!r.i64(v)) kv_ok = false;
            if (!kv_ok) break;
            if (f7[0] != k.format || f7[1] != cell_to) {
                kv_ok = false;
                break;
            }
            uint8_t mask = 0;
            if (!r.raw(&mask, 1)) {
                kv_ok = false;
                break;
            }
            ConversationBuffer* parts[5] = {&k.k, &k.v, &k.k_scale, &k.v_scale, &k.pooled};
            for (size_t part = 0; part < 5 && kv_ok; ++part) {
                uint64_t off = 0, len = 0;
                if (!r.u64(off) || !r.u64(len)) {
                    kv_ok = false;
                    break;
                }
                if ((mask & (1u << part)) == 0) continue;
                const uint64_t old_size = parts[part]->size();
                uint64_t new_size = 0;
                if (part == 4) {
                    if (seg_idx_block > 0 && seg_row_bytes > 0) {
                        const int64_t rows = live / seg_idx_block + 1;   // the pooled rows a full save holds at `live`
                        new_size = (uint64_t) rows * (uint64_t) seg_row_bytes;
                        k.pooled_rows = rows;
                    } else {
                        new_size = off + len;   // a v1 segment stored the block whole (v2 never writes one)
                    }
                } else {
                    if (k.cells <= 0 || old_size == 0 || old_size % (uint64_t) k.cells != 0) {
                        kv_ok = false;
                        break;
                    }
                    new_size = (old_size / (uint64_t) k.cells) * (uint64_t) cell_to;
                }
                if (off + len > new_size) {
                    kv_ok = false;
                    break;
                }
                parts[part]->resize_for_overwrite((size_t) new_size);
                uint64_t left = len, at_part = off;
                while (left && kv_ok) {
                    const size_t chunk = (size_t) std::min<uint64_t>(left, 16u << 20);
                    bool chunk_ok = true;
                    parts[part]->visit((size_t) at_part, chunk, [&](uint8_t* dst, size_t c, size_t) {
                        if (!r.raw(dst, c)) {
                            chunk_ok = false;
                            return false;
                        }
                        return true;
                    });
                    if (!chunk_ok) {
                        kv_ok = false;
                        break;
                    }
                    at_part += chunk;
                    left -= chunk;
                }
            }
        }
        if (!kv_ok) break;
        uint64_t cp_n = 0;
        if (!r.u64(cp_n) || cp_n != 0) break;   // routine segments carry none; a checkpoint segment is not applied
        uint64_t tail_hash = 0;
        const uint64_t expect = r.hash.digest();
        if (!r.plain(&tail_hash, 8)) break;
        if (tail_hash != expect) {
            error = "session delta: a segment failed its hash";
            break;
        }
        for (auto& k : image.kv) k.cells = cell_to;
        image.live.ids.insert(image.live.ids.end(), ids.begin(), ids.end());
        delta_used = true;
        at = r.at;
    }
    if (delta_used) delta_bytes = at - kDeltaHeaderBytes;
    return true;
}

// The base image with its increment applied: the convenience wrapper the disk tier's hit path uses.
inline bool session_delta_apply(const std::string& base_path, const std::string& delta_path,
                                const SessionFileIdentity& id, SavedConversation& image, size_t& bytes,
                                std::string& error, bool& delta_used, const SessionReadLimits& limits = {},
                                const SessionDeltaLimits& delta_limits = {}) {
    delta_used = false;
    bytes = 0;
    if (!session_file_read(base_path, id, image, bytes, error, limits)) return false;
    uint64_t delta_bytes = 0;
    return session_delta_overlay(base_path, delta_path, id, image, delta_bytes, error, delta_used, delta_limits);
}

// Drops a torn tail (structural check only: a segment that fits the file).  Returns the bytes that remain.
inline bool session_delta_truncate_to_valid(const std::string& delta_path, const SessionFileIdentity& id,
                                            int64_t base_tokens, uint64_t base_bytes, uint64_t& valid_bytes,
                                            std::string& error) {
    namespace d = conversation_delta_detail;
    valid_bytes = 0;
    std::vector<DeltaSegmentInfo> segs;
    std::string why;
    if (!session_delta_read(delta_path, id, base_tokens, base_bytes, segs, valid_bytes, why)) {
        error = why;
        return false;
    }
    std::FILE* f = std::fopen(delta_path.c_str(), "r+b");
    if (!f) return true;
    struct close_guard {
        std::FILE* f;
        ~close_guard() { std::fclose(f); }
    } guard{f};
    if (d::file_size_of(f) != valid_bytes && !d::truncate_to(f, valid_bytes)) {
        error = "session delta: cannot drop a torn tail";
        return false;
    }
    return true;
}

}   // namespace strata::core
