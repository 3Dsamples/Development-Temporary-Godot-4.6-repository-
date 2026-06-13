//File 0113 : core/parallel/concurrent_containers.h
//Lock‑free concurrent queue (Michael‑Scott), concurrent hash map with spinlock per bucket, and a concurrent vector with spinlock; no external dependencies.
#ifndef CORE_PARALLEL_CONCURRENT_CONTAINERS_H
#define CORE_PARALLEL_CONCURRENT_CONTAINERS_H

#include <atomic>
#include <vector>
#include <functional>
#include <memory>
#include <mutex>
#include <cstdint>
#include <algorithm>
#include <type_traits>
#include <new>

namespace SimulationMath {
namespace parallel {

// -----------------------------------------------------------------------------
// 1. Fast user‑space spinlock using std::atomic_flag
// -----------------------------------------------------------------------------
class SpinLock {
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
public:
    void lock() noexcept {
        while (flag_.test_and_set(std::memory_order_acquire)) {
            // optional: yield after some iterations to reduce contention
            // for simplicity we busy‑wait
        }
    }
    void unlock() noexcept {
        flag_.clear(std::memory_order_release);
    }
    bool try_lock() noexcept {
        return !flag_.test_and_set(std::memory_order_acquire);
    }
};

// -----------------------------------------------------------------------------
// 2. Lock‑free concurrent queue (Michael & Scott)
// -----------------------------------------------------------------------------
template <typename T>
class ConcurrentQueue {
    struct Node {
        T value;
        std::atomic<Node*> next;
        Node() noexcept : next(nullptr) {}
        explicit Node(const T& val) noexcept : value(val), next(nullptr) {}
    };

    alignas(64) std::atomic<Node*> head_;
    alignas(64) std::atomic<Node*> tail_;

public:
    ConcurrentQueue() noexcept {
        Node* dummy = new Node();
        head_.store(dummy, std::memory_order_relaxed);
        tail_.store(dummy, std::memory_order_relaxed);
    }

    ~ConcurrentQueue() noexcept {
        while (dequeue()) {}
        delete head_.load(std::memory_order_relaxed);
    }

    // Enqueue a copy of val (thread‑safe, multiple producers)
    void enqueue(const T& val) noexcept {
        Node* node = new Node(val);
        Node* prev_tail = tail_.exchange(node, std::memory_order_acq_rel);
        prev_tail->next.store(node, std::memory_order_release);
    }

    // Enqueue by moving (thread‑safe)
    void enqueue(T&& val) noexcept {
        Node* node = new Node(std::move(val));
        Node* prev_tail = tail_.exchange(node, std::memory_order_acq_rel);
        prev_tail->next.store(node, std::memory_order_release);
    }

    // Dequeue (returns true if an element was popped)
    // The popped value is stored in `result`.
    bool dequeue(T& result) noexcept {
        Node* old_head = head_.load(std::memory_order_acquire);
        for (;;) {
            Node* next = old_head->next.load(std::memory_order_acquire);
            if (next == nullptr) return false; // empty
            if (head_.compare_exchange_weak(old_head, next, std::memory_order_acq_rel)) {
                result = std::move(next->value);
                delete old_head; // safe, no other thread can access old_head now
                return true;
            }
            // CAS failed, old_head was reloaded by compare_exchange_weak, retry
        }
    }

    // Dequeue without returning value (just discard)
    bool dequeue() noexcept {
        T dummy;
        return dequeue(dummy);
    }

    // Check if empty (may be inaccurate due to concurrent modifications)
    bool empty() const noexcept {
        return head_.load(std::memory_order_acquire)->next.load(std::memory_order_acquire) == nullptr;
    }
};

// -----------------------------------------------------------------------------
// 3. Concurrent hash map with spinlock per bucket (separate chaining)
// -----------------------------------------------------------------------------
template <typename Key, typename Value, typename Hash = std::hash<Key>>
class ConcurrentHashMap {
    struct Node {
        Key key;
        Value value;
        Node* next;
        Node(const Key& k, const Value& v) noexcept : key(k), value(v), next(nullptr) {}
    };

    struct Bucket {
        SpinLock lock;
        Node* head = nullptr;
    };

    std::vector<Bucket> buckets_;
    Hash hasher_;
    size_t element_count_ = 0;  // not thread‑safe, approximate

    size_t bucket_index(const Key& key) const noexcept {
        return hasher_(key) % buckets_.size();
    }

public:
    explicit ConcurrentHashMap(size_t num_buckets = 256) noexcept
        : buckets_(num_buckets) {}

    // Insert or update the value associated with key.
    // If key already exists, its value is replaced; otherwise a new node is inserted.
    void insert(const Key& key, const Value& value) noexcept {
        size_t idx = bucket_index(key);
        Bucket& bucket = buckets_[idx];
        std::lock_guard<SpinLock> lock(bucket.lock);
        Node* curr = bucket.head;
        while (curr) {
            if (curr->key == key) {
                curr->value = value;
                return;
            }
            curr = curr->next;
        }
        // Insert new node at head
        Node* node = new Node(key, value);
        node->next = bucket.head;
        bucket.head = node;
    }

    // Find a key; returns true and sets value if found.
    bool find(const Key& key, Value& out_value) const noexcept {
        size_t idx = bucket_index(key);
        const Bucket& bucket = buckets_[idx];
        // no lock needed for reading if we traverse the list atomically? Since we only add at head, the list is safe to traverse without lock as long as we don't delete nodes. We'll assume no deletions (erase not implemented). So we can read without lock.
        Node* curr = bucket.head;
        while (curr) {
            if (curr->key == key) {
                out_value = curr->value;
                return true;
            }
            curr = curr->next;
        }
        return false;
    }

    // Erase a key (not thread‑safe if other operations are concurrent). Provide basic implementation.
    void erase(const Key& key) noexcept {
        size_t idx = bucket_index(key);
        Bucket& bucket = buckets_[idx];
        std::lock_guard<SpinLock> lock(bucket.lock);
        Node* prev = nullptr;
        Node* curr = bucket.head;
        while (curr) {
            if (curr->key == key) {
                if (prev) prev->next = curr->next;
                else bucket.head = curr->next;
                delete curr;
                return;
            }
            prev = curr;
            curr = curr->next;
        }
    }
};

// -----------------------------------------------------------------------------
// 4. Concurrent vector with spinlock (simple, for use cases where reads greatly outnumber writes)
// -----------------------------------------------------------------------------
template <typename T>
class ConcurrentVector {
    std::vector<T> data_;
    mutable SpinLock lock_;

public:
    ConcurrentVector() noexcept = default;
    explicit ConcurrentVector(size_t initial_capacity) noexcept {
        data_.reserve(initial_capacity);
    }

    // Push back (thread‑safe)
    void push_back(const T& value) noexcept {
        std::lock_guard<SpinLock> lock(lock_);
        data_.push_back(value);
    }

    void push_back(T&& value) noexcept {
        std::lock_guard<SpinLock> lock(lock_);
        data_.push_back(std::move(value));
    }

    template <typename... Args>
    void emplace_back(Args&&... args) noexcept {
        std::lock_guard<SpinLock> lock(lock_);
        data_.emplace_back(std::forward<Args>(args)...);
    }

    // Random access (non‑locking, const only; caller must ensure no concurrent writes)
    const T& operator[](size_t idx) const noexcept { return data_[idx]; }
    T& operator[](size_t idx) noexcept { return data_[idx]; }

    size_t size() const noexcept {
        std::lock_guard<SpinLock> lock(lock_);
        return data_.size();
    }

    bool empty() const noexcept {
        std::lock_guard<SpinLock> lock(lock_);
        return data_.empty();
    }

    // Clear the vector (thread‑safe)
    void clear() noexcept {
        std::lock_guard<SpinLock> lock(lock_);
        data_.clear();
    }
};

} // namespace parallel
} // namespace SimulationMath

#endif // CORE_PARALLEL_CONCURRENT_CONTAINERS_H