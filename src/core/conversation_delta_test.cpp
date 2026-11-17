// conversation_delta_test.cpp — the append-only increment beside a session file.
//
// S1 of the plan: write a base at N0 tokens, append segments, read the merged image back, and require it to be
// byte-for-byte what a full write at N would have read back as.  The page rounding of the K/V extent is what makes
// the arithmetic non-trivial, so the geometry here has cells != tokens.

#include "strata/core/conversation_delta.hpp"
#include "strata/core/conversation_disk_cache.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace strata::core;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", label);
        std::exit(1);
    }
}

constexpr int64_t kPage = 64;
constexpr int64_t kHeads = 2;
constexpr int64_t kHeadDim = 64;
constexpr int64_t kIdxDim = 32;
constexpr int64_t kIdxBlock = 512;   // tokens per pooled indexer row (row_bytes = kIdxDim * 4), as qsa_real_shapes

int64_t cells_of(int64_t tokens) { return ((tokens + kPage - 1) / kPage) * kPage; }

ConversationBuffer pattern(size_t n, uint8_t seed) {
    ConversationBuffer b;
    b.resize(n);
    b.visit(0, n, [&](uint8_t* p, size_t c, size_t at) {
        for (size_t i = 0; i < c; ++i) p[i] = uint8_t((at + i) * 131u + seed);
        return true;
    });
    return b;
}

std::vector<uint8_t> state_bytes(size_t n, uint8_t seed) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = uint8_t(i * 7u + seed);
    return v;
}

// One layer whose parts are the first `cells` cells of a fixed byte pattern: the bytes at a given offset never
// depend on the token count, so an image at N0 is a byte prefix of the image at N > N0 (the invariant the increment
// relies on).
SavedConversation image_at(int64_t tokens, uint8_t seed) {
    SavedConversation s;
    for (size_t i = 0; i < s.geometry.size(); ++i) s.geometry[i] = int64_t(100 + i);
    s.layer_lo = 0;
    s.layer_hi = 4;
    s.cvec = false;
    s.live.ids.resize((size_t) tokens);
    for (int64_t i = 0; i < tokens; ++i) s.live.ids[(size_t) i] = int32_t(1000 + i);
    s.live.imgs.clear();
    s.live.gdn = state_bytes(4099, seed);
    s.live.ple = state_bytes(77, uint8_t(seed + 1));
    s.live.tails = state_bytes(301, uint8_t(seed + 2));
    s.live.dead = state_bytes(64, uint8_t(seed + 3));
    s.live.block_pos = state_bytes(8, uint8_t(seed + 4));
    s.live.used = 42;

    const int64_t cells = cells_of(tokens);
    ConversationKv kv;
    kv.format = 2;
    kv.cells = cells;
    kv.heads = kHeads;
    kv.head_dim = kHeadDim;
    kv.page_size = kPage;
    kv.pooled_rows = tokens / kIdxBlock + 1;   // one indexer row per kIdxBlock tokens, as a full save holds them
    kv.idx_dim = kIdxDim;
    const size_t k_bytes = (size_t) cells * (size_t) kHeads * (size_t) kHeadDim;
    const size_t scale_bytes = (size_t) cells * (size_t) kHeads * (size_t) (kHeadDim / 64) * 2;
    kv.k = pattern(k_bytes, uint8_t(seed + 10));
    kv.v = pattern(k_bytes, uint8_t(seed + 20));
    kv.k_scale = pattern(scale_bytes, uint8_t(seed + 30));
    kv.v_scale = pattern(scale_bytes, uint8_t(seed + 40));
    kv.pooled = pattern((size_t) kv.pooled_rows * (size_t) kIdxDim * 4, uint8_t(seed + 50));
    s.kv.push_back(std::move(kv));
    return s;
}

bool read_into(const ConversationBuffer& b, size_t off, void* dst, size_t n) {
    if (off + n > b.size()) return false;
    auto* out = static_cast<uint8_t*>(dst);
    size_t done = 0;
    return b.visit(off, n, [&](const uint8_t* p, size_t c, size_t) {
        std::memcpy(out + done, p, c);
        done += c;
        return true;
    });
}

// the K/V part of `img` as sources, the way the engine hands the disk a live slot's bytes (device in production)
std::vector<SessionKvSource> sources_of(const SavedConversation& img, std::string& error) {
    std::vector<SessionKvSource> out(img.kv.size());
    for (size_t i = 0; i < img.kv.size(); ++i) {
        const ConversationKv& k = img.kv[i];
        SessionKvSource s;
        s.format = k.format;
        s.cells = k.cells;
        s.heads = k.heads;
        s.head_dim = k.head_dim;
        s.page_size = k.page_size;
        s.pooled_rows = k.pooled_rows;
        s.idx_dim = k.idx_dim;
        s.idx_block = kIdxBlock;
        s.sizes = {k.k.size(), k.v.size(), k.k_scale.size(), k.v_scale.size(), k.pooled.size()};
        const ConversationBuffer* parts[5] = {&k.k, &k.v, &k.k_scale, &k.v_scale, &k.pooled};
        auto why = std::make_shared<std::string>();
        s.read = [parts, why](size_t part, size_t offset, void* dst, size_t n) {
            if (part >= 5 || !read_into(*parts[part], offset, dst, n)) {
                *why = "read out of range";
                return false;
            }
            return true;
        };
        s.error = [why] { return *why; };
        out[i] = std::move(s);
    }
    (void) error;
    return out;
}

bool same_part(const ConversationBuffer& a, const ConversationBuffer& b) {
    if (a.size() != b.size()) return false;
    std::vector<uint8_t> x(a.size()), y(b.size());
    if (!a.empty() && !read_into(a, 0, x.data(), x.size())) return false;
    if (!b.empty() && !read_into(b, 0, y.data(), y.size())) return false;
    return x == y;
}

bool same_image(const SavedConversation& a, const SavedConversation& b) {
    if (a.live.ids != b.live.ids || a.live.gdn != b.live.gdn || a.live.ple != b.live.ple ||
        a.live.tails != b.live.tails || a.live.dead != b.live.dead || a.live.block_pos != b.live.block_pos)
        return false;
    if (a.kv.size() != b.kv.size()) return false;
    for (size_t i = 0; i < a.kv.size(); ++i) {
        const auto& x = a.kv[i];
        const auto& y = b.kv[i];
        if (x.format != y.format || x.cells != y.cells || x.heads != y.heads || x.head_dim != y.head_dim ||
            x.page_size != y.page_size || x.pooled_rows != y.pooled_rows || x.idx_dim != y.idx_dim)
            return false;
        if (!same_part(x.k, y.k) || !same_part(x.v, y.v) || !same_part(x.k_scale, y.k_scale) ||
            !same_part(x.v_scale, y.v_scale) || !same_part(x.pooled, y.pooled))
            return false;
    }
    return true;
}

// a full write of `img` at `path`, read back: the reference the merged image must equal
bool full_round_trip(const fs::path& path, const SavedConversation& img, const SessionFileIdentity& id,
                     SavedConversation& back, std::string& error) {
    size_t written = 0;
    if (!session_file_write(path.string(), img, id, written, error)) return false;
    size_t read = 0;
    return session_file_read(path.string(), id, back, read, error);
}

SavedConversation without_kv(SavedConversation img) {
    img.kv.clear();
    return img;
}
}   // namespace

int main() {
    std::fprintf(stderr, "step: start\n"); std::fflush(stderr);
    // %TEMP% on this host refuses create_directories (the process dies with 0xC0000409), so the harness works in the
    // build folder it is started from; the in-tree test overrides this with STRATA_DELTA_TEST_DIR.
    const char* env_dir = std::getenv("STRATA_DELTA_TEST_DIR");
    const fs::path dir = env_dir ? fs::path(env_dir) : fs::current_path() / "delta-test-work";
    fs::remove_all(dir);
    std::error_code mk;
    fs::create_directories(dir, mk);
    check(!mk, "the test folder exists");
    const SessionFileIdentity id{0x1111222233334444ull, 0x5555666677778888ull};
    std::string error;

    std::fprintf(stderr, "step: dir made\n"); std::fflush(stderr);
    const int64_t n0 = 4100, n1 = 4608, n2 = 5120;
    const SavedConversation base = image_at(n0, 1);
    const SavedConversation at1 = image_at(n1, 1);
    const SavedConversation at2 = image_at(n2, 1);

    const fs::path base_path = dir / "conv.conv";
    const std::string delta_path = session_delta_path(base_path.string());

    size_t base_bytes = 0;
    std::fprintf(stderr, "step: images built\n"); std::fflush(stderr);
    check(session_file_write(base_path.string(), base, id, base_bytes, error), "base written");
    std::fprintf(stderr, "step: base written\n"); std::fflush(stderr);
    const uint64_t base_file = (uint64_t) fs::file_size(base_path);
    check(base_file == base_bytes, "base size matches the report");

    // ---- segment 1: [n0, n1)
    {
        auto sources = sources_of(at1, error);
        size_t bytes = 0;
        const bool seg1 = session_delta_append(delta_path, without_kv(at1), sources, id, n0, base_file, n0, bytes, error);
        if (!seg1) std::fprintf(stderr, "segment 1 error: %s\n", error.c_str());
        check(seg1, "segment 1 appended");
        check(bytes < base_file / 4, "segment 1 is a fraction of the base (the tail, not the image)");
        std::vector<DeltaSegmentInfo> segs;
        uint64_t valid = 0;
        check(session_delta_read(delta_path, id, n0, base_file, segs, valid, error), "segment 1 readable");
        check(segs.size() == 1 && segs[0].live_tokens == n1 && segs[0].prev_live_tokens == n0, "segment 1 chain");
        check(segs[0].cell_from == (n0 / kPage) * kPage, "segment 1 starts on a page boundary");
        check(segs[0].cell_to == cells_of(n1), "segment 1 ends on the padded extent of its token count");
    }

    // ---- the merged image equals a full write at n1, byte for byte
    {
        SavedConversation merged;
        size_t bytes = 0;
        bool used = false;
        check(session_delta_apply(base_path.string(), delta_path, id, merged, bytes, error, used),
              "merge over the base succeeds");
        check(used, "the increment was applied");
        SavedConversation reference;
        check(full_round_trip(dir / "full1.conv", at1, id, reference, error), "reference write at n1");
        check(same_image(merged, reference), "merged image == full write at n1 (byte for byte)");
    }

    // ---- segment 2: [n1, n2) on top of segment 1
    {
        auto sources = sources_of(at2, error);
        size_t bytes = 0;
        check(session_delta_append(delta_path, without_kv(at2), sources, id, n0, base_file, n1, bytes, error),
              "segment 2 appended");
        SavedConversation merged;
        size_t read = 0;
        bool used = false;
        check(session_delta_apply(base_path.string(), delta_path, id, merged, read, error, used), "merge 1+2");
        check(used, "both segments applied");
        SavedConversation reference;
        check(full_round_trip(dir / "full2.conv", at2, id, reference, error), "reference write at n2");
        check(same_image(merged, reference), "merged image == full write at n2 (byte for byte)");
        std::vector<DeltaSegmentInfo> segs;
        uint64_t valid = 0;
        check(session_delta_read(delta_path, id, n0, base_file, segs, valid, error) && segs.size() == 2,
              "two segments listed");
    }

    // ---- a torn tail costs the tail only: truncate four bytes off and the earlier segments still merge
    {
        const uint64_t full = (uint64_t) fs::file_size(delta_path);
        fs::resize_file(delta_path, full - 4);
        std::vector<DeltaSegmentInfo> segs;
        uint64_t valid = 0;
        check(session_delta_read(delta_path, id, n0, base_file, segs, valid, error), "torn tail still readable");
        check(segs.size() == 1, "the torn segment is not listed");
        SavedConversation merged;
        size_t read = 0;
        bool used = false;
        check(session_delta_apply(base_path.string(), delta_path, id, merged, read, error, used), "merge on a torn file");
        check(used, "segment 1 still applied");
        SavedConversation reference;
        check(full_round_trip(dir / "full1b.conv", at1, id, reference, error) && same_image(merged, reference),
              "the torn file merges to n1 exactly");
        uint64_t trimmed = 0;
        check(session_delta_truncate_to_valid(delta_path, id, n0, base_file, trimmed, error), "truncate to valid");
        check((uint64_t) fs::file_size(delta_path) == trimmed, "the torn tail is gone");
        // and appending works again after the heal
        auto sources = sources_of(at2, error);
        size_t bytes = 0;
        check(session_delta_append(delta_path, without_kv(at2), sources, id, n0, base_file, n1, bytes, error),
              "append after the heal");
        SavedConversation again;
        size_t read2 = 0;
        bool used2 = false;
        check(session_delta_apply(base_path.string(), delta_path, id, again, read2, error, used2) && used2 &&
                  same_image(again, [&] {
                      SavedConversation r;
                      std::string e;
                      full_round_trip(dir / "full2b.conv", at2, id, r, e);
                      return r;
                  }()),
              "healed increment merges to n2 exactly");
    }

    // ---- an increment that belongs to another base is not applied (and says why)
    {
        SavedConversation merged;
        size_t read = 0;
        bool used = false;
        std::string why;
        check(session_delta_apply(base_path.string(), delta_path, id, merged, read, why, used) && used,
              "sanity: it does apply to its own base");
        std::vector<DeltaSegmentInfo> segs;
        uint64_t valid = 0;
        std::string bad;
        check(!session_delta_read(delta_path, id, n0 + 1, base_file, segs, valid, bad), "another base is refused");
        check(!bad.empty(), "the refusal says why");
    }

    // ---- no increment at all: the base stands, nothing is applied
    {
        SavedConversation merged;
        size_t read = 0;
        bool used = true;
        check(session_delta_apply(base_path.string(), session_delta_path((dir / "nope.conv").string()), id, merged,
                                  read, error, used),
              "a missing increment is not an error");
        check(!used, "nothing was applied");
        check(same_image(merged, base), "the image is the base itself");
    }

    // ---- the header of a stale increment (other model) is refused
    {
        const SessionFileIdentity other{0x9999000011112222ull, 0x5555666677778888ull};
        std::vector<DeltaSegmentInfo> segs;
        uint64_t valid = 0;
        std::string bad;
        check(!session_delta_read(delta_path, other, n0, base_file, segs, valid, bad), "another model is refused");
    }

    // ---- the idle flush (--conversation-cache-disk-flush-sec): a parked conversation reaches the disk from its own
    // buffers, and what the disk then holds merges to exactly a full write of the grown conversation
    {
        const fs::path dir2 = dir / "idle";
        fs::create_directories(dir2, mk);
        ConversationDiskCache disk(dir2.string(), (uint64_t) 1 << 30, 2048, 0);
        disk.set_identity([](SessionFileIdentity& out, std::string&) {
            out = SessionFileIdentity{0x1111222233334444ull, 0x5555666677778888ull};
            return true;
        });
        ConversationCache ram((size_t) 1 << 30, 4, 2048);
        disk.attach(&ram);
        disk.enable_delta(1, 0, 0);   // the flush's own thresholds; the fold off
        check(disk.ready(), "the disk tier is wired to a RAM cache");

        // the base, the way an eviction hands an image over
        SavedConversation evicted = base;
        disk.enqueue(std::move(evicted));
        disk.flush(std::vector<SavedConversation>{}, 30);
        check(disk.stats().writes == 1, "the eviction wrote the base");

        SavedConversation grown = image_at(n2, 1);
        check(ram.put(std::move(grown)), "the grown conversation is parked");
        uint64_t wrote = 0;
        const size_t seen = ram.flush_visit(3, [&](const SavedConversation& img) {
            // the shipped builder reads the same five parts of the parked buffers (conversation_kv_image_sources,
            // which lives in conversation_snapshot.cpp with the kernels); sources_of is the same read here
            std::vector<SessionKvSource> src = sources_of(img, error);
            wrote += disk.flush_image(img, 256, std::move(src));
        });
        check(seen == 1, "the idle pass visited the parked conversation");
        check(wrote > 0, "the idle pass wrote its growth");
        check(disk.stats().delta_writes == 1 && disk.stats().flushes == 1, "one segment, counted as an idle flush");
        uint64_t again1 = 0;
        const size_t seen2 = ram.flush_visit(3, [&](const SavedConversation& img) {
            std::vector<SessionKvSource> src = sources_of(img, error);
            again1 += disk.flush_image(img, 256, std::move(src));
        });
        check(seen2 == 1 && again1 == 0 && disk.stats().flushes == 1, "a second pass has nothing new to write");

        // what the disk now holds, read back as a reader would: base + chain == a full write at n2
        fs::path written;
        for (const auto& e : fs::directory_iterator(dir2))
            if (e.path().extension() == ".conv") written = e.path();
        check(!written.empty(), "the base file is where the disk tier keeps it");
        SavedConversation merged;
        size_t read = 0;
        bool used = false;
        check(session_delta_apply(written.string(), session_delta_path(written.string()), id, merged, read, error,
                                  used) && used,
              "the idle flush's chain applies over the base");
        SavedConversation reference;
        check(full_round_trip(dir / "full-idle.conv", image_at(n2, 1), id, reference, error), "reference write at n2");
        check(same_image(merged, reference), "the idle flush leaves exactly a full image at n2");
        disk.close();
    }

    // ---- a parked conversation the disk has never seen: the idle pass writes the whole image (the one a hard kill
    // would lose), and the pass after it has nothing to do
    {
        const fs::path dir3 = dir / "idle-new";
        fs::create_directories(dir3, mk);
        ConversationDiskCache disk(dir3.string(), (uint64_t) 1 << 30, 2048, 0);
        disk.set_identity([](SessionFileIdentity& out, std::string&) {
            out = SessionFileIdentity{0x1111222233334444ull, 0x5555666677778888ull};
            return true;
        });
        ConversationCache ram((size_t) 1 << 30, 4, 2048);
        disk.attach(&ram);
        disk.enable_delta(1, 0, 0);

        SavedConversation fresh = image_at(n0, 7);
        for (auto& t : fresh.live.ids) t += 500000;   // another conversation, so another file
        check(ram.put(std::move(fresh)), "the unseen conversation is parked");
        uint64_t wrote = 0;
        ram.flush_visit(3, [&](const SavedConversation& img) {
            std::vector<SessionKvSource> src = sources_of(img, error);
            wrote += disk.flush_image(img, 256, std::move(src));
        });
        check(wrote > 0, "the whole image went to the disk");
        check(disk.stats().writes == 1 && disk.stats().flushes == 1, "counted as a write and as an idle flush");
        uint64_t again = 0;
        ram.flush_visit(3, [&](const SavedConversation& img) {
            std::vector<SessionKvSource> src = sources_of(img, error);
            again += disk.flush_image(img, 256, std::move(src));
        });
        check(again == 0 && disk.stats().writes == 1, "the pass after it writes nothing");

        // and a request for it finds it on the disk again (the recall path a restart uses).  recall() matches a
        // PROMPT that continues the parked conversation, so probe with one more token, as the next request does.
        const SavedConversation& parked = *ram.find(1);
        std::vector<int32_t> probe = parked.live.ids;
        probe.push_back(777777);
        check(disk.recall(probe, parked.live.imgs, parked.cvec), "recall reads the idle write back");
        check(ram.size() == 2, "the recalled image is parked beside the one it came from");
        disk.close();
    }

    std::printf("conversation_delta_test: %d checks passed\n", checks);
    fs::remove_all(dir);
    return 0;
}
