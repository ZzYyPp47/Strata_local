// Conversation disk tier: a third home for parked conversations, below VRAM and the RAM cache.
//
// What it does: a parked conversation that the RAM cache pushes out (or that is still parked at QUIT) is written to
// one file per conversation under --conversation-cache-disk DIR; the next request for that conversation finds the file
// by a token-prefix fingerprint, reads it back and puts it into the RAM cache, so the restore path that already exists
// (conversation_cache -> conversation_snapshot_restore_prefix) takes over instead of a cold prefill.
//
// 显存活跃槽 K/V -> 内存会话缓存（--conversation-cache-mib）-> 磁盘（这里）。The disk tier itself never touches the
// device: it only moves SavedConversation images between RAM and files.
//
// File format: the engine's own session file (conversation_file.hpp), written and read with the same identity binding
// (model files x resolved config), so every built-in guard applies unchanged: a regular single-link file, the size, the
// header, the identity, a bounded parse and the payload hash.  A file that fails any of them is a miss, never a fatal.
//
// Index: "<file>.idx", one line of key=value, beside the file.  It is a hint (which conversation, how long, when last
// used), never the authority: the payload's own validation decides whether a file is usable.
//
// Quota: --conversation-cache-disk-gb N (default 700, 0 = unlimited).  Over the cap, the files used longest ago go
// first (last_used_at = max(written, hit)).  A single image larger than the cap is skipped.
//
// Crash safety: session_file_write's temporary file + flush + rename; a half-written file never carries an index, and
// an index without a valid payload is dropped on the next read.  The constructor sweeps leftovers.
//
// Default off: no --conversation-cache-disk means no thread, no file, no behaviour change.

#ifndef STRATA_CORE_CONVERSATION_DISK_CACHE_HPP
#define STRATA_CORE_CONVERSATION_DISK_CACHE_HPP

#include <strata/core/conversation_cache.hpp>
#include <strata/core/conversation_delta.hpp>
#include <strata/core/conversation_file.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace strata::core {

// The first tokens that name a conversation.  Stable across turns (a conversation only appends), long enough that two
// unrelated conversations do not share it, short enough that the key costs nothing.
inline constexpr size_t kConvDiskKeyTokens = 256;
// How many conversations may share one root prefix (system prompt) before the oldest file is replaced.
inline constexpr size_t kConvDiskSlots = 4;

struct ConvDiskStats {
    uint64_t files = 0, bytes = 0, hits = 0, misses = 0, writes = 0, write_errors = 0, dropped = 0, quota_deletes = 0,
             sweep_deletes = 0, refused_put = 0;
    double last_write_ms = 0, last_read_ms = 0;
    // the increment tier (--conversation-cache-disk-delta): segments appended, their bytes, compactions, failures
    uint64_t delta_writes = 0, delta_bytes = 0, delta_errors = 0, compactions = 0, delta_reads = 0;
    double last_delta_ms = 0;
    uint64_t last_delta_read_bytes = 0;
    // the idle flusher (--conversation-cache-disk-flush-sec): images written while the engine sat quiet
    uint64_t flushes = 0, flush_bytes = 0;
    uint64_t last_flush_bytes = 0;
    double last_flush_ms = 0;
    // The skip screen (STRATA_CONV_DISK_SKIP): a slot that has been read whole and judged "not this prompt's prefix"
    // `skip_min_miss` times in a row (conversation_prefix() found neither its live state nor any checkpoint a strict
    // prefix of the prompt) has its whole-file read skipped instead of paid for nothing.  The name `skipped_long` is
    // historical: the screen no longer tests the file's length, since a file shorter than the prompt is the case that
    // keeps getting read back for nothing.  `skip_retries` is how often such a slot was read again after
    // STRATA_CONV_DISK_SKIP_RETRY_S, `reclaimed` how many such files were deleted after STRATA_CONV_DISK_RECLAIM_AFTER_S
    // of never being hit.
    uint64_t skipped_long = 0, skip_retries = 0, reclaimed = 0;
};

// Where one conversation's increment stands: the base file it extends and the newest point on its chain.
struct ConvDiskDeltaInfo {
    int64_t base_tokens = 0;    // the base file's token count (fixed until a compaction rewrites it)
    int64_t saved_tokens = 0;   // the newest live point a segment (or the base) covers
    int64_t prev_live = 0;      // what the next segment chains from
    uint64_t base_bytes = 0;
    int64_t at_ms = 0;          // when the last segment was written (0: never)
    std::string key, path;      // which conversation, and which file of it
};

class ConversationDiskCache {
public:
    using IdentityFn = std::function<bool(SessionFileIdentity& id, std::string& error)>;

    ConversationDiskCache() = default;
    ConversationDiskCache(std::string dir, uint64_t cap_bytes, int64_t min_tokens, uint64_t min_free_bytes)
        : dir_(std::move(dir)), cap_bytes_(cap_bytes), min_tokens_(min_tokens), min_free_bytes_(min_free_bytes) {
        if (dir_.empty()) return;
        // The screen's knobs, read once at start-up: STRATA_CONV_DISK_SKIP=0 turns all of it off.
        if (const char* e = std::getenv("STRATA_CONV_DISK_SKIP"); e != nullptr && std::string(e) == "0") skip_enabled_ = false;
        if (const char* e = std::getenv("STRATA_CONV_DISK_SKIP_MIN_MISS"); e != nullptr)
            skip_min_miss_ = std::max<int64_t>(1, std::atoll(e));
        if (const char* e = std::getenv("STRATA_CONV_DISK_SKIP_RETRY_S"); e != nullptr)
            skip_retry_ms_ = std::max<int64_t>(0, std::atoll(e)) * 1000;
        if (const char* e = std::getenv("STRATA_CONV_DISK_RECLAIM_AFTER_S"); e != nullptr)
            reclaim_after_s_ = std::max<int64_t>(0, std::atoll(e));
        enabled_ = true;
        sweep();
        writer_ = std::thread([this] { write_loop(); });
    }
    ~ConversationDiskCache() { close(); }
    ConversationDiskCache(const ConversationDiskCache&) = delete;
    ConversationDiskCache& operator=(const ConversationDiskCache&) = delete;

    bool enabled() const { return enabled_; }
    const std::string& dir() const { return dir_; }
    uint64_t cap_bytes() const { return cap_bytes_; }
    ConvDiskStats stats() const {
        std::lock_guard<std::mutex> lk(mu_);
        return stats_;
    }

    // The identity of the engine's model + resolved config: computed once by the caller's lambda (it samples the model
    // files).  Called from the writer thread and from a hit, so it must be thread safe (generate.cpp caches it).
    void set_identity(IdentityFn fn) { identity_ = std::move(fn); }
    // Where an evicted image goes.  Wired by generate.cpp once the cache exists.
    void attach(ConversationCache* cache) { cache_ = cache; }
    bool ready() const { return enabled_ && identity_ && cache_ != nullptr; }

    // Read the newest `count` files into RAM now, so the first request after a restart does not wait for the disk.
    // Background thread: startup is idle time.  Bounded (a prewarmed image is held outside the RAM cache's budget
    // until a request takes it), so it is one or two images, never a directory.
    void prewarm(int count);

    // ---- the increment tier (--conversation-cache-disk-delta) ------------------------------------------------
    // A conversation that grew by a turn is appended to `<file>.delta` as the byte tail of its K/V, so the write
    // costs that turn's bytes instead of the whole image (the reader merges the two, see conversation_delta.hpp).
    void enable_delta(int64_t min_tokens, int64_t sec, int compact_pct);
    bool delta_enabled() const { return delta_; }

    // What a turn-end hands the writer: the live state (small) and read sources over the live K/V (device or pinned
    // host pool, so no host copy of the image is made).  The sources stay valid because the caller joins before it
    // reuses the slot they read from.
    struct DeltaJob {
        std::string key, path;
        int64_t base_tokens = 0, prev_live = 0;
        uint64_t base_bytes = 0;
        SavedConversation meta;
        std::shared_ptr<const std::vector<SessionKvSource>> kv;
    };

    // Should this conversation's increment be written now?  `out` is the chain the new segment extends.
    template <class Token>
    bool delta_due(const std::vector<Token>& ids, const std::vector<ConversationImageKey>& imgs, bool cvec,
                   int64_t live_tokens, ConvDiskDeltaInfo& out, int64_t min_growth = -1,
                   bool respect_time = true, bool check_queue = true);

    void delta_submit(DeltaJob&& job);

    bool delta_busy() const {
        std::lock_guard<std::mutex> lk(mu_);
        return delta_writing_ || !delta_queue_.empty();
    }
    // Waits until the increments in flight are on the disk: their sources point into a slot that is about to change.
    void delta_join();

    // Try the disk for this conversation.  On a hit the image is put into the RAM cache (`cache_`), so the caller's
    // next best() finds it; a miss is silent.  Never throws, never fatal.
    template <class Token>
    bool recall(const std::vector<Token>& ids, const std::vector<ConversationImageKey>& imgs, bool cvec) {
        if (!ready() || ids.empty()) return false;
        SessionFileIdentity id;
        std::string err;
        if (!identity_(id, err)) return false;
        const std::string key = make_key(ids, imgs, cvec, id);
        if (key.empty()) return false;
        const auto t0 = std::chrono::steady_clock::now();
        // A prewarmed image was read at startup (prewarm()): use it exactly like a file just read from the disk.
        for (size_t slot = 0; slot < kConvDiskSlots; ++slot) {
            const std::string path = path_of_key(key, slot);
            SavedConversation warm;
            {
                std::lock_guard<std::mutex> lk(mu_);
                for (size_t i = 0; i < warm_.size(); ++i)
                    if (warm_[i].first == path) {
                        warm = std::move(warm_[i].second);
                        warm_.erase(warm_.begin() + (std::ptrdiff_t) i);
                        break;
                    }
            }
            if (warm.live.ids.empty()) continue;
            bool warm_ok = conversation_prefix(warm.live, ids, imgs) > 0;
            if (!warm_ok)
                for (const auto& c : warm.checkpoints)
                    if (conversation_prefix(c, ids, imgs) > 0) {
                        warm_ok = true;
                        break;
                    }
            if (!warm_ok || warm.stage_images.size() > 1) continue;
            if (!cache_->put(std::move(warm))) continue;
            note_read(true, t0, path);
            return true;
        }
        for (size_t slot = 0; slot < kConvDiskSlots; ++slot) {
            const std::string path = path_of_key(key, slot);
            Idx idx;
            if (!read_idx(path, idx)) continue;
            if (idx.model != id.model || idx.config != id.config || idx.cvec != (cvec ? 1 : 0)) continue;
            // The skip screen: once a slot has been read whole and judged "not this prompt's prefix" skip_min_miss times
            // in a row, its multi-GiB read is skipped instead of paid for nothing.  A length test is deliberately NOT a
            // condition here: a file SHORTER than the prompt is exactly the case that keeps getting read back for
            // nothing (a file already no shorter than the prompt can never be its prefix either, so both sides need the
            // same screen).  The retry window keeps a slot from being skipped for good, and a hit forgets it entirely.
            if (skip_enabled_) {
                bool skip = false;
                {
                    std::lock_guard<std::mutex> lk(mu_);
                    SkipNote& note = skips_[path];
                    const int64_t ms = now_ms();
                    if (note.skipping && ms - note.at_ms >= skip_retry_ms_) {
                        note.skipping = false;
                        ++stats_.skip_retries;
                    } else if (note.skipping || note.miss >= (uint32_t) skip_min_miss_) {
                        note.skipping = true;
                        note.at_ms = ms;
                        ++stats_.skipped_long;
                        skip = true;
                    }
                }
                if (skip) continue;
            }
            SavedConversation image;
            size_t bytes = 0;
            SessionStatus st;
            std::string error;
            if (!session_file_read(path, id, image, bytes, error, {}, &st)) {
                // a stale or damaged file: its index goes, the file follows on the next sweep
                std::error_code ec;
                std::filesystem::remove(idx_path_of(path), ec);
                std::fprintf(stderr, "strata serve: conversation disk: stale file dropped (%s): %s [%s]\n",
                             std::filesystem::path(path).filename().string().c_str(), error.c_str(),
                             session_error_name(st.error));
                note_read(false, t0, path);
                note_slot_miss(path);
                continue;
            }
            // The increment beside it (--conversation-cache-disk-delta) is applied on top of the base: the image is
            // then exactly what a full write of that conversation would have read back as.
            overlay_delta(path, id, image);
            // A usable image is one whose live state OR one of whose checkpoints is a strict token prefix of this
            // prompt.  A checkpoint is the point: the file was written after a turn whose reply the client may render
            // differently, so the live state's tail often differs while a checkpoint at 16 K/32 K/48 K tokens still
            // matches exactly - the matcher (ConversationCache::best/borrows) then restores that prefix and only the
            // rest is read.  Anything else is not this conversation: drop the index and go on.
            bool usable = conversation_prefix(image.live, ids, imgs) > 0;
            if (!usable)
                for (const auto& c : image.checkpoints)
                    if (conversation_prefix(c, ids, imgs) > 0) {
                        usable = true;
                        break;
                    }
            if (!usable) {
                if (std::getenv("STRATA_L2_TRACE") != nullptr) {
                    const size_t lv = image.live.ids.size(), pr = ids.size();
                    size_t common = 0;
                    while (common < lv && common < pr && image.live.ids[common] == ids[common]) ++common;
                    std::fprintf(stderr, "strata l2: recall miss %s: live %zu prompt %zu common %zu ckpts %zu\n",
                                 std::filesystem::path(path).filename().string().c_str(), lv, pr, common,
                                 image.checkpoints.size());
                }
                note_read(false, t0, path);
                note_slot_miss(path);
                reclaim_if_idle(path, idx);
                continue;   // another conversation that shares our first tokens
            }
            if (image.stage_images.size() > 1) continue;   // a layer split writes one image per stage: only stage 0
            touch_idx(path, idx);
            if (!cache_->put(std::move(image))) {
                std::lock_guard<std::mutex> lk(mu_);
                ++stats_.refused_put;
                ++stats_.misses;
                return false;
            }
            note_read(true, t0, path);
            note_slot_hit(path);
            return true;
        }
        note_read(false, t0, std::string());
        return false;
    }

    // An evicted image (drop_oldest's hook).  Moves it to the writer; never blocks on the disk.
    void enqueue(SavedConversation&& image) {
        if (!enabled_ || !identity_) return;
        if ((int64_t) image.live.ids.size() < min_tokens_) return;
        if (image.stage_images.size() > 1) return;   // layer split: not one file per conversation
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (pending_.size() >= 2) {
                ++stats_.dropped;
                return;   // a backlog means the disk is the bottleneck; the RAM cache is the truth for now
            }
            pending_.push_back(std::move(image));
        }
        cv_.notify_one();
    }

    // QUIT: hand over what the RAM cache still holds and wait (bounded) for the writer.
    void flush(std::vector<SavedConversation>&& images, int seconds) {
        if (!enabled_) return;
        delta_join();
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (auto& img : images) {
                if ((int64_t) img.live.ids.size() < min_tokens_ || img.stage_images.size() > 1) continue;
                pending_.push_back(std::move(img));
            }
            flushing_ = true;
        }
        cv_.notify_one();
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
        std::unique_lock<std::mutex> lk(mu_);
        cv_done_.wait_until(lk, deadline, [this] { return pending_.empty() && !writing_; });
    }

    // The engine log's one line about the disk tier.
    std::string summary() const {
        std::lock_guard<std::mutex> lk(mu_);
        char b[640];
        std::snprintf(b, sizeof b,
                      "disk: %llu files %.1f GiB / %.1f GiB cap, hits %llu miss %llu writes %llu (last %.0f ms) "
                      "errors %llu dropped %llu quota-deleted %llu, skipped %llu retries %llu reclaimed %llu, "
                      "delta %llu segments %.2f GiB (+%.0f ms last, "
                      "read %llu) compactions %llu errors %llu, idle-flush %llu %.2f GiB (last %.2f GiB %.0f ms)",
                      (unsigned long long) stats_.files, (double) stats_.bytes / 1073741824.0,
                      cap_bytes_ == 0 ? 0.0 : (double) cap_bytes_ / 1073741824.0, (unsigned long long) stats_.hits,
                      (unsigned long long) stats_.misses, (unsigned long long) stats_.writes, stats_.last_write_ms,
                      (unsigned long long) stats_.write_errors, (unsigned long long) stats_.dropped,
                      (unsigned long long) stats_.quota_deletes, (unsigned long long) stats_.skipped_long,
                      (unsigned long long) stats_.skip_retries, (unsigned long long) stats_.reclaimed,
                      (unsigned long long) stats_.delta_writes,
                      (double) stats_.delta_bytes / 1073741824.0, stats_.last_delta_ms,
                      (unsigned long long) stats_.delta_reads, (unsigned long long) stats_.compactions,
                      (unsigned long long) stats_.delta_errors, (unsigned long long) stats_.flushes,
                      (double) stats_.flush_bytes / 1073741824.0, (double) stats_.last_flush_bytes / 1073741824.0,
                      stats_.last_flush_ms);
        return std::string(b);
    }

    void close() {
        if (!enabled_) return;
        enabled_ = false;
        {
            std::lock_guard<std::mutex> lk(mu_);
            stopping_ = true;
        }
        cv_.notify_all();
        cv_delta_.notify_all();
        if (writer_.joinable()) writer_.join();
        if (warms_.joinable()) warms_.join();
        if (delta_writer_.joinable()) delta_writer_.join();
    }

private:
    struct Idx {
        int64_t tokens = 0, at = 0;
        uint64_t model = 0, config = 0, bytes = 0;
        int cvec = 1;
        bool ok() const { return tokens > 0; }
    };

    std::string path_of_key(const std::string& key, size_t slot) const;

    template <class Token>
    std::string make_key(const std::vector<Token>& ids, const std::vector<ConversationImageKey>& imgs, bool cvec,
                         const SessionFileIdentity& id) const {
        const size_t n = std::min(ids.size(), kConvDiskKeyTokens);
        if (n == 0) return std::string();
        SessionIdentityBuilder b(0x5eed5eed5eedULL);
        b.u64("model", id.model);
        b.u64("config", id.config);
        b.i64("cvec", cvec ? 1 : 0);
        b.i64("n", (int64_t) n);
        for (size_t i = 0; i < n; ++i) b.i64("t", (int64_t) ids[i]);
        for (const auto& img : imgs)
            if (img.start < (int64_t) n) b.i64("i", img.start);
        char out[17];
        std::snprintf(out, sizeof out, "%016llx", (unsigned long long) b.digest());
        return std::string(out);
    }
    static std::string idx_path_of(const std::string& path);
    bool read_idx(const std::string& path, Idx& idx) const;
    void write_idx(const std::string& path, const Idx& idx) const;
    void touch_idx(const std::string& path, Idx idx) const;
    // Applies the increment beside `path` onto an image read from it (idempotent: a segment already covered is
    // skipped).  Every path that turns a base file into an image calls it - the read, the startup prewarm and a
    // prewarmed image taken later - or a prewarmed conversation would silently lose what its increment had added.
    void overlay_delta(const std::string& path, const SessionFileIdentity& id, SavedConversation& image,
                       uint64_t* added = nullptr);
    void note_read(bool hit, std::chrono::steady_clock::time_point t0, const std::string& path);
    void sweep();
    void enforce_quota();
    void write_loop();
    bool write_one(const SavedConversation& image, uint64_t* written = nullptr);
    void delta_loop();
    bool delta_write_one(DeltaJob& job);

public:
    // The idle flusher's write (generate.cpp's conv_flusher): the conversation a park left in the RAM cache, written
    // straight out of its own host buffers.  Appends the part the chain does not cover when a base exists and the
    // growth passes `min_growth`; writes the whole image when the disk holds no base for this conversation at all.
    // Returns the bytes written (0: nothing to do, the chain is current, or the write was refused).  The caller holds
    // the engine's conversation scope, so no park or eviction can move these buffers under the write.
    uint64_t flush_image(const SavedConversation& image, int64_t min_growth,
                         std::vector<SessionKvSource>&& sources);

private:
    static int64_t now_ms() {
        return (int64_t) std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    }
    static int64_t now_s() {
        return (int64_t) std::chrono::duration_cast<std::chrono::seconds>(
                   std::chrono::system_clock::now().time_since_epoch())
            .count();
    }
    // The skip screen's bookkeeping: a miss resets the retry clock and clears a skip; a hit forgets the slot (its
    // file was just replaced or is useful again).
    void note_slot_miss(const std::string& path) {
        std::lock_guard<std::mutex> lk(mu_);
        SkipNote& note = skips_[path];
        ++note.miss;
        note.at_ms = now_ms();
        note.skipping = false;
    }
    void note_slot_hit(const std::string& path) {
        std::lock_guard<std::mutex> lk(mu_);
        skips_.erase(path);
    }
    // A slot nobody has hit for STRATA_CONV_DISK_RECLAIM_AFTER_S and that keeps missing is a file whose conversation
    // has not come back: it holds one of the four slots this key has, and write_one refuses to replace a base older
    // than its increment, so without this it would be re-read for nothing forever.  idx.at is only ever refreshed by a
    // hit (touch_idx), which is what makes "old" mean "unused".
    void reclaim_if_idle(const std::string& path, const Idx& idx) {
        if (!skip_enabled_ || reclaim_after_s_ <= 0 || idx.at <= 0) return;
        if (now_s() - idx.at < reclaim_after_s_) return;
        {
            std::lock_guard<std::mutex> lk(mu_);
            auto it = skips_.find(path);
            if (it == skips_.end() || it->second.miss < (uint32_t) skip_min_miss_) return;
        }
        std::error_code e1, e2;
        std::filesystem::remove(path, e1);
        std::filesystem::remove(idx_path_of(path), e2);
        {
            std::lock_guard<std::mutex> lk(mu_);
            skips_.erase(path);
            ++stats_.reclaimed;
        }
        std::fprintf(stderr, "strata serve: conversation disk: stale file reclaimed (%s, %lld tokens, idle %lld s)\n",
                     std::filesystem::path(path).filename().string().c_str(), (long long) idx.tokens,
                     (long long) (now_s() - idx.at));
    }

    std::string dir_;
    uint64_t cap_bytes_ = 0, min_free_bytes_ = 0;
    int64_t min_tokens_ = 0;
    // The skip screen and the reclaim of files nobody has hit for long (see ConvDiskStats' comment).
    struct SkipNote {
        int64_t at_ms = 0;      // when the slot was last looked at (steady clock)
        uint32_t miss = 0;      // misses in a row in this process
        bool skipping = false;  // its whole-file read is being skipped right now
    };
    std::unordered_map<std::string, SkipNote> skips_;
    bool skip_enabled_ = true;
    int64_t skip_min_miss_ = 3, skip_retry_ms_ = 600000, reclaim_after_s_ = 259200;
    bool enabled_ = false;
    mutable std::mutex mu_;
    std::condition_variable cv_, cv_done_;
    std::deque<SavedConversation> pending_;
    std::thread writer_;
    bool stopping_ = false, writing_ = false, flushing_ = false;
    ConvDiskStats stats_;
    IdentityFn identity_;
    ConversationCache* cache_ = nullptr;
    // prewarm(): images read at startup, kept parsed (outside the RAM cache's budget) until a request takes one
    std::vector<std::pair<std::string, SavedConversation>> warm_;
    std::thread warms_;
    bool warm_started_ = false;
    // the increment tier: its own thread, its own queue, and where each conversation's chain stands
    bool delta_ = false, delta_started_ = false, delta_writing_ = false;
    int64_t delta_min_tokens_ = 1024, delta_sec_ = 60;
    int delta_compact_pct_ = 20;
    std::deque<DeltaJob> delta_queue_;
    std::thread delta_writer_;
    std::condition_variable cv_delta_, cv_delta_done_;
    std::unordered_map<std::string, ConvDiskDeltaInfo> deltas_;
};

// ---- implementation (header-only) ----

inline std::string ConversationDiskCache::idx_path_of(const std::string& path) { return path + ".idx"; }

inline std::string ConversationDiskCache::path_of_key(const std::string& key, size_t slot) const {
    return (std::filesystem::path(dir_) / (key + "-" + std::to_string(slot) + ".conv")).string();
}

inline void ConversationDiskCache::overlay_delta(const std::string& path, const SessionFileIdentity& id,
                                                 SavedConversation& image, uint64_t* added) {
    if (added != nullptr) *added = 0;
    if (!delta_ || image.live.ids.empty()) return;
    uint64_t delta_bytes = 0;
    bool used = false;
    std::string why;
    session_delta_overlay(path, session_delta_path(path), id, image, delta_bytes, why, used);
    if (used) {
        if (added != nullptr) *added = delta_bytes;
        std::lock_guard<std::mutex> lk(mu_);
        ++stats_.delta_reads;
        stats_.last_delta_read_bytes = delta_bytes;
    } else if (!why.empty()) {
        // An increment this build cannot apply stays where it is: an image already carrying the chain reads as
        // "another base file", and removing it there would throw away a good chain (the append self-heals a torn one).
        std::fprintf(stderr, "strata serve: conversation disk: delta not applied (%s): %s\n",
                     std::filesystem::path(path).filename().string().c_str(), why.c_str());
    }
}

inline bool ConversationDiskCache::read_idx(const std::string& path, Idx& idx) const {
    std::ifstream in(idx_path_of(path), std::ios::binary);
    if (!in) return false;
    std::string line;
    std::getline(in, line);
    Idx o;
    size_t p = 0;
    while (p < line.size()) {
        const size_t eq = line.find('=', p);
        if (eq == std::string::npos) break;
        const size_t sp = line.find(' ', eq);
        const std::string k = line.substr(p, eq - p);
        const std::string v = line.substr(eq + 1, (sp == std::string::npos ? line.size() : sp) - eq - 1);
        try {
            if (k == "tokens") o.tokens = std::stoll(v);
            else if (k == "at") o.at = std::stoll(v);
            else if (k == "model") o.model = std::stoull(v);
            else if (k == "config") o.config = std::stoull(v);
            else if (k == "bytes") o.bytes = std::stoull(v);
            else if (k == "cvec") o.cvec = std::stoi(v);
        } catch (...) {
            return false;
        }
        if (sp == std::string::npos) break;
        p = sp + 1;
    }
    if (!o.ok()) return false;
    idx = o;
    return true;
}

inline void ConversationDiskCache::write_idx(const std::string& path, const Idx& idx) const {
    const std::string ip = idx_path_of(path);
    const std::string tmp = ip + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) return;
        out << "tokens=" << idx.tokens << " model=" << idx.model << " config=" << idx.config << " cvec=" << idx.cvec
            << " bytes=" << idx.bytes << " at=" << idx.at << "\n";
        out.flush();
        if (!out) return;
    }
    std::error_code ec;
    std::filesystem::remove(ip, ec);
    std::filesystem::rename(tmp, ip, ec);
}

inline void ConversationDiskCache::touch_idx(const std::string& path, Idx idx) const {
    idx.at = (int64_t) std::chrono::duration_cast<std::chrono::seconds>(
                  std::chrono::system_clock::now().time_since_epoch())
                  .count();
    write_idx(path, idx);
}

inline void ConversationDiskCache::note_read(bool hit, std::chrono::steady_clock::time_point t0,
                                            const std::string& path) {
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::lock_guard<std::mutex> lk(mu_);
    if (hit) ++stats_.hits; else ++stats_.misses;
    stats_.last_read_ms = ms;
    if (hit)
        std::fprintf(stderr, "strata serve: conversation disk: hit %.0f ms (%s)\n", ms,
                     std::filesystem::path(path).filename().string().c_str());
    else if (!path.empty())
        std::fprintf(stderr, "strata serve: conversation disk: stale file dropped (%s)\n",
                     std::filesystem::path(path).filename().string().c_str());
}

inline void ConversationDiskCache::sweep() {
    std::error_code ec;
    std::filesystem::create_directories(dir_, ec);
    if (ec) {
        std::fprintf(stderr, "strata serve: conversation disk off: cannot create %s (%s)\n", dir_.c_str(),
                     ec.message().c_str());
        enabled_ = false;
        return;
    }
    uint64_t total = 0, files = 0, swept = 0;
    for (auto it = std::filesystem::directory_iterator(dir_, ec); !ec && it != std::filesystem::directory_iterator();
         it.increment(ec)) {
        const auto& p = it->path();
        const std::string ext = p.extension().string();
        std::error_code e2;
        const auto sz = (uint64_t) it->file_size(e2);
        if (ext == ".tmp") {
            std::filesystem::remove(p, e2);   // a write that never got to rename
            ++swept;
            continue;
        }
        if (ext == ".idx") continue;
        if (ext == ".delta") {
            // an increment whose base is gone is unreadable by design
            const std::string s = p.string();
            const std::string base = s.substr(0, s.size() - std::string(".delta").size());
            if (!std::filesystem::exists(base, e2)) {
                std::filesystem::remove(p, e2);
                ++swept;
                continue;
            }
            total += sz;
            continue;
        }
        if (ext != ".conv") continue;
        if (!std::filesystem::exists(idx_path_of(p.string()), e2)) {
            std::filesystem::remove(p, e2);   // no index: unreadable by design
            ++swept;
            continue;
        }
        total += sz;
        ++files;
    }
    {
        std::lock_guard<std::mutex> lk(mu_);
        stats_.bytes = total;
        stats_.files = files;
        stats_.sweep_deletes = swept;
    }
    std::fprintf(stderr,
                 "strata serve: conversation disk: %s (%.1f GiB in %llu files, %.1f GiB cap, swept %llu)\n", dir_.c_str(),
                 (double) total / 1073741824.0, (unsigned long long) files,
                 cap_bytes_ == 0 ? 0.0 : (double) cap_bytes_ / 1073741824.0, (unsigned long long) swept);
    enforce_quota();
}

// Over the cap the files used longest ago go first.  The just-written file is never the victim of its own write.
inline void ConversationDiskCache::enforce_quota() {
    if (cap_bytes_ == 0) return;
    for (int round = 0; round < 4096; ++round) {
        struct Victim {
            std::string path;
            int64_t at = 0;
            uint64_t bytes = 0;
        };
        Victim v;
        std::vector<Victim> all;
        uint64_t total = 0;
        std::error_code ec;
        for (auto it = std::filesystem::directory_iterator(dir_, ec);
             !ec && it != std::filesystem::directory_iterator(); it.increment(ec)) {
            if (it->path().extension() != ".conv") continue;
            Idx idx;
            std::error_code e2;
            uint64_t sz = (uint64_t) it->file_size(e2);
            if (!read_idx(it->path().string(), idx)) continue;
            std::error_code e4;
            const uint64_t dsz = (uint64_t) std::filesystem::file_size(session_delta_path(it->path().string()), e4);
            if (!e4) sz += dsz;   // the increment counts against the cap like the base
            all.push_back({it->path().string(), idx.at, sz});
            total += sz;
        }
        if (total <= cap_bytes_) {
            std::lock_guard<std::mutex> lk(mu_);
            stats_.bytes = total;
            stats_.files = all.size();
            return;
        }
        if (all.empty()) return;
        v = all.front();
        for (const auto& c : all)
            if (c.at < v.at) v = c;
        std::error_code e3;
        std::filesystem::remove(v.path, e3);
        std::filesystem::remove(idx_path_of(v.path), e3);
        std::filesystem::remove(session_delta_path(v.path), e3);
        std::fprintf(stderr, "strata serve: conversation disk: over cap, deleted %s (%.1f GiB)\n",
                     std::filesystem::path(v.path).filename().string().c_str(), (double) v.bytes / 1073741824.0);
        std::lock_guard<std::mutex> lk(mu_);
        ++stats_.quota_deletes;
    }
}

inline void ConversationDiskCache::write_loop() {
    for (;;) {
        SavedConversation image;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [this] { return stopping_ || !pending_.empty(); });
            if (stopping_ && pending_.empty()) return;
            image = std::move(pending_.front());
            pending_.pop_front();
            writing_ = true;
        }
        write_one(image);
        image = SavedConversation();
        {
            std::lock_guard<std::mutex> lk(mu_);
            writing_ = false;
        }
        cv_done_.notify_all();
    }
}

inline uint64_t ConversationDiskCache::flush_image(const SavedConversation& image, int64_t min_growth,
                                                   std::vector<SessionKvSource>&& sources) {
    static const bool fl_trace = std::getenv("STRATA_DELTA_TRACE") != nullptr;
    if (!ready()) {
        if (fl_trace) std::fprintf(stderr, "strata flush image: disk tier not ready\n");
        return 0;
    }
    const int64_t tokens = (int64_t) image.live.ids.size();
    // (2026-11-18) No second min-tokens gate here: flush_visit already applied `wants()` to this entry, and the
    // flusher's own growth gate decides the rest.  A duplicate gate silently dropped every conversation between
    // the two thresholds.
    if (tokens <= 0) return 0;
    SessionFileIdentity id;
    std::string err;
    if (!identity_(id, err)) {
        if (fl_trace) std::fprintf(stderr, "strata flush image: identity unavailable: %s\n", err.c_str());
        return 0;
    }
    // Claim the increment writer for the whole pass: it is what delta_join() waits on and what a turn-end coalesces on,
    // and it keeps two writers off one chain.  Released on every way out.
    struct Claim {
        ConversationDiskCache* self;
        bool taken = false;
        explicit Claim(ConversationDiskCache* c) : self(c) {
            std::lock_guard<std::mutex> lk(self->mu_);
            if (self->delta_writing_ || !self->delta_queue_.empty()) return;
            self->delta_writing_ = true;
            taken = true;
        }
        ~Claim() {
            if (!taken) return;
            {
                std::lock_guard<std::mutex> lk(self->mu_);
                self->delta_writing_ = false;
            }
            self->cv_delta_done_.notify_all();
        }
    } claim{this};
    if (!claim.taken) {
        if (fl_trace) std::fprintf(stderr, "strata flush image: a delta writer is busy\n");
        return 0;
    }
    const auto t0 = std::chrono::steady_clock::now();
    uint64_t bytes = 0;
    ConvDiskDeltaInfo info;
    if (!sources.empty() && delta_ &&
        delta_due(image.live.ids, image.live.imgs, image.cvec, tokens, info, min_growth, false, false)) {
        // Only the part the base does not cover goes, read straight out of the parked buffers: the live state and the
        // deepest checkpoint are what the segment needs besides them (a copy of those, not of the K/V).
        SavedConversation meta;
        meta.geometry = image.geometry;
        meta.layer_lo = image.layer_lo;
        meta.layer_hi = image.layer_hi;
        meta.cvec = image.cvec;
        meta.live = image.live;
        meta.checkpoints = session_checkpoints_to_save(image.checkpoints);
        DeltaJob job;
        job.key = info.key;
        job.path = info.path;
        job.base_tokens = info.base_tokens;
        job.prev_live = info.prev_live;
        job.base_bytes = info.base_bytes;
        job.meta = std::move(meta);
        job.kv = std::make_shared<const std::vector<SessionKvSource>>(std::move(sources));
        DeltaJob running = std::move(job);
        uint64_t before = 0;
        {
            std::lock_guard<std::mutex> lk(mu_);
            before = stats_.delta_bytes;
        }
        if (!delta_write_one(running)) {
            if (fl_trace) std::fprintf(stderr, "strata flush image: delta write refused\n");
            return 0;
        }
        {
            std::lock_guard<std::mutex> lk(mu_);
            bytes = stats_.delta_bytes > before ? stats_.delta_bytes - before : 0;
        }
        if (fl_trace) std::fprintf(stderr, "strata flush image: delta %llu bytes\n", (unsigned long long) bytes);
    } else {
        // Nothing to append: either the chain is already current, or the disk holds no base for this conversation at
        // all - and that one a hard kill loses outright, so the whole image goes now.
        const std::string key = make_key(image.live.ids, image.live.imgs, image.cvec, id);
        if (key.empty()) {
            if (fl_trace) std::fprintf(stderr, "strata flush image: empty key\n");
            return 0;
        }
        for (size_t slot = 0; slot < kConvDiskSlots; ++slot) {
            Idx idx;
            const std::string ipath = path_of_key(key, slot);
            if (!read_idx(ipath, idx)) continue;
            if (fl_trace)
                std::fprintf(stderr, "strata flush image: idx %s model %llu/%llu config %llu/%llu cvec %u/%u\n",
                             ipath.c_str(), (unsigned long long) idx.model, (unsigned long long) id.model,
                             (unsigned long long) idx.config, (unsigned long long) id.config, (unsigned) idx.cvec,
                             (unsigned) (image.cvec ? 1 : 0));
            if (idx.model != id.model || idx.config != id.config || idx.cvec != (image.cvec ? 1 : 0)) continue;
            if (fl_trace) std::fprintf(stderr, "strata flush image: the disk already holds it\n");
            return 0;   // the disk already holds this conversation, or a chain that extends it
        }
        const bool wrote_whole = write_one(image, &bytes);
        if (fl_trace)
            std::fprintf(stderr, "strata flush image: whole write ok=%d bytes=%llu\n", (int) wrote_whole,
                         (unsigned long long) bytes);
        if (!wrote_whole) return 0;
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::lock_guard<std::mutex> lk(mu_);
    ++stats_.flushes;
    stats_.flush_bytes += bytes;
    stats_.last_flush_bytes = bytes;
    stats_.last_flush_ms = ms;
    return bytes;
}

inline bool ConversationDiskCache::write_one(const SavedConversation& image, uint64_t* written) {
    if (!identity_) return false;
    SessionFileIdentity id;
    std::string err;
    if (!identity_(id, err)) return false;
    const std::string key = make_key(image.live.ids, image.live.imgs, image.cvec, id);
    if (key.empty()) return false;
    // The same conversation always lands in the same slot (its whole token list names it); a different conversation
    // that shares the root prefix takes the next one, and wraps onto the oldest when all are taken.
    uint64_t full = 1469598103934665603ULL;
    for (const auto& t : image.live.ids) {
        full ^= (uint64_t) (int64_t) t;
        full *= 1099511628211ULL;
    }
    const size_t slot = (size_t) (full % kConvDiskSlots);
    const std::string path = path_of_key(key, slot);
    {
        // A rewritten base must never be older than the increment beside it: that would lose the newer tail.  The
        // newer chain stays, the stale image is dropped (the RAM cache still has it).
        std::lock_guard<std::mutex> lk(mu_);
        auto it = deltas_.find(key);
        if (it != deltas_.end() && it->second.saved_tokens > (int64_t) image.live.ids.size()) {
            ++stats_.dropped;
            return false;
        }
    }
    const auto t0 = std::chrono::steady_clock::now();
    SessionWriteOptions wo;
    wo.min_free_bytes = min_free_bytes_;
    size_t bytes = 0;
    SessionStatus st;
    const bool ok = session_file_write(path, image, id, bytes, err, wo, &st);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (!ok) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            ++stats_.write_errors;
        }
        std::fprintf(stderr, "strata serve: conversation disk: write failed (%s): %s\n", path.c_str(), err.c_str());
        return false;
    }
    Idx idx;
    idx.tokens = (int64_t) image.live.ids.size();
    idx.model = id.model;
    idx.config = id.config;
    idx.cvec = image.cvec ? 1 : 0;
    idx.bytes = bytes;
    idx.at = (int64_t) std::chrono::duration_cast<std::chrono::seconds>(
                  std::chrono::system_clock::now().time_since_epoch())
                  .count();
    write_idx(path, idx);
    {
        std::lock_guard<std::mutex> lk(mu_);
        ++stats_.writes;
        stats_.last_write_ms = ms;
        // a fresh file must not inherit the misses of the one it replaced
        skips_.erase(path);
    }
    std::fprintf(stderr, "strata serve: conversation disk: wrote %lld tokens, %.2f GiB in %.0f ms (%s)\n",
                 (long long) idx.tokens, (double) bytes / 1073741824.0, ms,
                 std::filesystem::path(path).filename().string().c_str());
    if (delta_) {
        // the base now covers at least what the increment did: it is redundant (the write is itself a compaction)
        std::error_code ec;
        std::filesystem::remove(session_delta_path(path), ec);
        std::lock_guard<std::mutex> lk(mu_);
        deltas_.erase(key);
    }
    if (written != nullptr) *written = bytes;
    enforce_quota();
    return true;
}

// Startup is the one idle window in which the disk can be read for free: the newest files are parsed here, so the
// first request after a restart finds its image already in RAM and pays no disk read at all.
inline void ConversationDiskCache::prewarm(int count) {
    if (!enabled_ || !identity_ || count <= 0) return;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (warm_started_) return;
        warm_started_ = true;
    }
    warms_ = std::thread([this, count] {
        SessionFileIdentity id;
        std::string err;
        if (!identity_(id, err)) return;   // no identity yet: the next start warms instead
        struct Cand {
            std::string path;
            int64_t at;
            uint64_t bytes;
        };
        std::vector<Cand> cand;
        std::error_code ec;
        for (std::filesystem::directory_iterator it(dir_, ec), end; !ec && it != end; it.increment(ec)) {
            if (it->path().extension() != ".conv") continue;
            const std::string p = it->path().string();
            Idx idx;
            if (!read_idx(p, idx)) continue;
            if (idx.model != id.model || idx.config != id.config) continue;
            cand.push_back({p, idx.at, idx.bytes});
        }
        std::sort(cand.begin(), cand.end(), [](const Cand& a, const Cand& b) { return a.at > b.at; });
        if ((int) cand.size() > count) cand.resize((size_t) count);
        size_t got = 0;
        for (const auto& c : cand) {
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (stopping_) return;
            }
            SavedConversation image;
            size_t bytes = 0;
            SessionStatus st;
            std::string error;
            const auto t0 = std::chrono::steady_clock::now();
            if (!session_file_read(c.path, id, image, bytes, error, {}, &st)) continue;
            uint64_t delta_added = 0;
            overlay_delta(c.path, id, image, &delta_added);   // the base alone is not this conversation
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (stopping_) return;
                warm_.emplace_back(c.path, std::move(image));
                while (warm_.size() > (size_t) count) warm_.erase(warm_.begin());
            }
            ++got;
            std::fprintf(stderr, "strata serve: conversation disk: prewarmed %.2f GiB in %.0f ms (%s)\n",
                         (double) (bytes + delta_added) / 1073741824.0, ms,
                         std::filesystem::path(c.path).filename().string().c_str());
        }
        if (got > 0)
            std::fprintf(stderr, "strata serve: conversation disk: prewarmed %zu of %zu file(s) before the first request\n",
                         got, cand.size());
    });
}

// ---- the increment tier: turn-end triggers, one writer thread, the 20% compaction rule ----------------------------
inline void ConversationDiskCache::enable_delta(int64_t min_tokens, int64_t sec, int compact_pct) {
    std::lock_guard<std::mutex> lk(mu_);
    delta_ = enabled_;
    if (min_tokens > 0) delta_min_tokens_ = min_tokens;
    if (sec > 0) delta_sec_ = sec;
    delta_compact_pct_ = compact_pct;   // 0 turns the fold off: the chain then folds at the next full write
    if (delta_ && !delta_started_) {
        delta_started_ = true;
        delta_writer_ = std::thread([this] { delta_loop(); });
    }
}

template <class Token>
bool ConversationDiskCache::delta_due(const std::vector<Token>& ids, const std::vector<ConversationImageKey>& imgs,
                                     bool cvec, int64_t live_tokens, ConvDiskDeltaInfo& out, int64_t min_growth,
                                     bool respect_time, bool check_queue) {
    if (!delta_ || !enabled_ || !identity_) return false;
    SessionFileIdentity id;
    std::string err;
    if (!identity_(id, err)) return false;
    const std::string key = make_key(ids, imgs, cvec, id);
    if (key.empty()) return false;
    const int64_t now = now_ms();
    std::lock_guard<std::mutex> lk(mu_);
    if (check_queue && (delta_writing_ || !delta_queue_.empty())) return false;   // one in flight: the next turn-end coalesces
    auto it = deltas_.find(key);
    if (it == deltas_.end()) {
        ConvDiskDeltaInfo info;
        for (size_t slot = 0; slot < kConvDiskSlots; ++slot) {
            const std::string path = path_of_key(key, slot);
            Idx idx;
            if (!read_idx(path, idx)) continue;
            if (idx.model != id.model || idx.config != id.config || idx.cvec != (cvec ? 1 : 0)) continue;
            std::error_code ec;
            const uint64_t bytes = (uint64_t) std::filesystem::file_size(path, ec);
            if (ec) continue;
            info.base_tokens = idx.tokens;
            info.saved_tokens = idx.tokens;
            info.prev_live = idx.tokens;
            info.base_bytes = bytes;
            info.at_ms = 0;   // never written: the first segment may go as soon as there is enough new
            // An increment written before this process started (a restart, a hard kill) still extends this base: the
            // next segment starts where that chain ends, not at the base's own token count, or every append after a
            // restart would be refused as "newer segments than this save assumes".
            {
                std::vector<DeltaSegmentInfo> segs;
                uint64_t valid = 0;
                std::string why;
                const bool ok = session_delta_read(session_delta_path(path), id, idx.tokens, bytes, segs, valid, why);
                if (!ok && !why.empty()) {
                    // A chain this build cannot read (another format, or its base moved): write no segment for it
                    // rather than let the append drop a chain that may still be good.
                    if (std::getenv("STRATA_DELTA_TRACE") != nullptr)
                        std::fprintf(stderr, "strata serve: conversation disk: delta chain unreadable (%s): %s\n",
                                     key.c_str(), why.c_str());
                    return false;
                }
                if (!segs.empty()) {
                    info.saved_tokens = segs.back().live_tokens;
                    info.prev_live = segs.back().live_tokens;
                }
                if (std::getenv("STRATA_DELTA_TRACE") != nullptr)
                    std::fprintf(stderr, "strata serve: conversation disk: delta chain %s: %zu segment(s), saved %lld "
                                         "(base %lld, %llu bytes)\n",
                                 key.c_str(), segs.size(), (long long) info.saved_tokens, (long long) idx.tokens,
                                 (unsigned long long) bytes);
            }
            info.key = key;
            info.path = path;
            break;
        }
        if (info.base_tokens <= 0) {
            if (std::getenv("STRATA_DELTA_TRACE") != nullptr)
                std::fprintf(stderr, "strata serve: conversation disk: delta has no base for %s\n", key.c_str());
            return false;   // nothing on the disk to extend yet (the first full write comes first)
        }
        it = deltas_.emplace(key, info).first;
    }
    ConvDiskDeltaInfo& st = it->second;
    st.key = key;
    if (std::getenv("STRATA_DELTA_TRACE") != nullptr)
        std::fprintf(stderr, "strata serve: conversation disk: delta due check %s: live %lld base %lld saved %lld min %lld at_ms %lld\n",
                     key.c_str(), (long long) live_tokens, (long long) st.base_tokens, (long long) st.saved_tokens,
                     (long long) delta_min_tokens_, (long long) st.at_ms);
    if (live_tokens - st.saved_tokens < (min_growth >= 0 ? min_growth : delta_min_tokens_)) return false;
    if (respect_time && st.at_ms != 0 && now - st.at_ms < delta_sec_ * 1000) return false;
    out = st;
    return true;
}

inline void ConversationDiskCache::delta_submit(DeltaJob&& job) {
    if (!delta_ || !job.kv) return;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (delta_queue_.size() >= 2) return;   // the disk is the bottleneck: this turn's tail will be in the next one
        delta_queue_.push_back(std::move(job));
    }
    cv_delta_.notify_one();
}

inline void ConversationDiskCache::delta_join() {
    std::unique_lock<std::mutex> lk(mu_);
    if (!delta_started_) return;
    cv_delta_done_.wait(lk, [this] { return !delta_writing_ && delta_queue_.empty(); });
}

inline void ConversationDiskCache::delta_loop() {
    for (;;) {
        DeltaJob job;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_delta_.wait(lk, [this] { return stopping_ || !delta_queue_.empty(); });
            if (stopping_ && delta_queue_.empty()) return;
            job = std::move(delta_queue_.front());
            delta_queue_.pop_front();
            delta_writing_ = true;
        }
        delta_write_one(job);
        {
            std::lock_guard<std::mutex> lk(mu_);
            delta_writing_ = false;
        }
        cv_delta_done_.notify_all();
    }
}

inline bool ConversationDiskCache::delta_write_one(DeltaJob& job) {
    static const bool dtrace = std::getenv("STRATA_DELTA_TRACE") != nullptr;
    if (dtrace) std::fprintf(stderr, "strata serve: conversation disk: delta writer took %s\n", job.key.c_str());
    if (!identity_ || !job.kv || job.kv->empty()) {
        if (dtrace) std::fprintf(stderr, "strata serve: conversation disk: delta writer: no sources\n");
        return false;
    }
    SessionFileIdentity id;
    std::string err;
    if (!identity_(id, err)) return false;
    const int64_t live = (int64_t) job.meta.live.ids.size();
    if (live <= job.prev_live) {
        if (dtrace) std::fprintf(stderr, "strata serve: conversation disk: delta writer: nothing new (%lld <= %lld)\n",
                                 (long long) live, (long long) job.prev_live);
        return false;
    }
    const std::string delta_path = session_delta_path(job.path);
    const auto t0 = std::chrono::steady_clock::now();
    SessionWriteOptions wo;
    wo.min_free_bytes = 0;   // a segment is small: the base's own reserve already passed
    size_t bytes = 0;
    std::string error;
    const bool ok = session_delta_append(delta_path, job.meta, *job.kv, id, job.base_tokens, job.base_bytes,
                                         job.prev_live, bytes, error, wo);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (!ok) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            ++stats_.delta_errors;
        }
        std::fprintf(stderr, "strata serve: conversation disk: delta failed (%s): %s\n",
                     std::filesystem::path(job.path).filename().string().c_str(), error.c_str());
        return false;
    }
    uint64_t chain_bytes = 0;
    {
        std::vector<DeltaSegmentInfo> segs;
        uint64_t valid = 0;
        std::string why;
        session_delta_read(delta_path, id, job.base_tokens, job.base_bytes, segs, valid, why);
        chain_bytes = valid;
    }
    {
        std::lock_guard<std::mutex> lk(mu_);
        ++stats_.delta_writes;
        stats_.delta_bytes += bytes;
        stats_.last_delta_ms = ms;
        ConvDiskDeltaInfo& st = deltas_[job.key];
        st.key = job.key;
        st.path = job.path;
        st.base_tokens = job.base_tokens;
        st.base_bytes = job.base_bytes;
        st.saved_tokens = live;
        st.prev_live = live;
        st.at_ms = now_ms();
    }
    std::fprintf(stderr, "strata serve: conversation disk: delta %lld tokens (+%.2f GiB in %.0f ms, chain %.2f GiB) (%s)\n",
                 (long long) (live - job.prev_live), (double) bytes / 1073741824.0, ms,
                 (double) chain_bytes / 1073741824.0,
                 std::filesystem::path(job.path).filename().string().c_str());

    // The fold: past a percentage of the base (and 32 MiB) the chain is rewritten into a fresh base from the same
    // sources.  A setting of 0 turns the fold off: the chain then simply keeps growing and the next full write (QUIT,
    // or the RAM cache pushing the conversation out) is what folds it - that path is the proven one.
    const uint64_t cap = delta_compact_pct_ <= 0
                             ? 0
                             : std::max<uint64_t>(32u << 20, job.base_bytes / 100u * (uint64_t) delta_compact_pct_);
    if (cap == 0 || chain_bytes <= cap) return true;
    const auto c0 = std::chrono::steady_clock::now();
    size_t nbytes = 0;
    std::string cerr;
    SessionStatus cst;
    if (!session_file_write(job.path, job.meta, *job.kv, id, nbytes, cerr, wo, &cst)) {
        std::lock_guard<std::mutex> lk(mu_);
        ++stats_.delta_errors;
        return true;   // the segment is on the disk; the compaction waits for the next turn
    }
    std::error_code ec;
    std::filesystem::remove(delta_path, ec);
    Idx idx;
    idx.tokens = live;
    idx.model = id.model;
    idx.config = id.config;
    idx.cvec = job.meta.cvec ? 1 : 0;
    idx.bytes = nbytes;
    idx.at = (int64_t) std::chrono::duration_cast<std::chrono::seconds>(
                  std::chrono::system_clock::now().time_since_epoch())
                  .count();
    write_idx(job.path, idx);
    const double cms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c0).count();
    {
        std::lock_guard<std::mutex> lk(mu_);
        ++stats_.compactions;
        ConvDiskDeltaInfo& st = deltas_[job.key];
        st.base_tokens = live;
        st.saved_tokens = live;
        st.prev_live = live;
        st.base_bytes = nbytes;
        st.at_ms = now_ms();
    }
    std::fprintf(stderr,
                 "strata serve: conversation disk: compacted %s (%.2f GiB base, %.2f GiB chain) in %.0f ms\n",
                 std::filesystem::path(job.path).filename().string().c_str(), (double) nbytes / 1073741824.0,
                 (double) chain_bytes / 1073741824.0, cms);
    return true;
}

} // namespace strata::core

#endif
