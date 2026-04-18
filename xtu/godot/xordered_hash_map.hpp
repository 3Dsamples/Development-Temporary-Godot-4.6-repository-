// include/xtu/godot/xordered_hash_map.hpp
// xtensor-unified - Ordered hash map container for Godot 4.6
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XORDERED_HASH_MAP_HPP
#define XTU_GODOT_XORDERED_HASH_MAP_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "xtu/core/xtensor_config.hpp"

XTU_NAMESPACE_BEGIN
namespace godot {

// #############################################################################
// OrderedHashMap - Hash map that preserves insertion order
// #############################################################################
template <typename K, typename V, typename Hash = std::hash<K>, typename KeyEqual = std::equal_to<K>>
class OrderedHashMap {
public:
    using Key = K;
    using Value = V;
    using Pair = std::pair<Key, Value>;
    using ListIterator = typename std::list<Pair>::iterator;
    using ConstListIterator = typename std::list<Pair>::const_iterator;

private:
    std::list<Pair> m_list;
    std::unordered_map<Key, ListIterator, Hash, KeyEqual> m_map;

public:
    OrderedHashMap() = default;
    
    OrderedHashMap(std::initializer_list<Pair> init) {
        for (const auto& p : init) {
            insert(p.first, p.second);
        }
    }

    // Capacity
    size_t size() const { return m_map.size(); }
    bool empty() const { return m_map.empty(); }

    // Insertion
    void insert(const Key& key, const Value& value) {
        auto it = m_map.find(key);
        if (it != m_map.end()) {
            // Update existing
            it->second->second = value;
        } else {
            // Insert new at end
            m_list.emplace_back(key, value);
            m_map[key] = std::prev(m_list.end());
        }
    }

    void insert(Key&& key, Value&& value) {
        auto it = m_map.find(key);
        if (it != m_map.end()) {
            it->second->second = std::move(value);
        } else {
            m_list.emplace_back(std::move(key), std::move(value));
            m_map[std::move(key)] = std::prev(m_list.end());
        }
    }

    template <typename... Args>
    void emplace(const Key& key, Args&&... args) {
        auto it = m_map.find(key);
        if (it != m_map.end()) {
            it->second->second = Value(std::forward<Args>(args)...);
        } else {
            m_list.emplace_back(key, Value(std::forward<Args>(args)...));
            m_map[key] = std::prev(m_list.end());
        }
    }

    // Access
    Value& operator[](const Key& key) {
        auto it = m_map.find(key);
        if (it != m_map.end()) {
            return it->second->second;
        }
        m_list.emplace_back(key, Value());
        auto list_it = std::prev(m_list.end());
        m_map[key] = list_it;
        return list_it->second;
    }

    const Value& operator[](const Key& key) const {
        auto it = m_map.find(key);
        if (it == m_map.end()) {
            throw std::out_of_range("Key not found in OrderedHashMap");
        }
        return it->second->second;
    }

    Value& at(const Key& key) {
        auto it = m_map.find(key);
        if (it == m_map.end()) {
            throw std::out_of_range("Key not found in OrderedHashMap");
        }
        return it->second->second;
    }

    const Value& at(const Key& key) const {
        auto it = m_map.find(key);
        if (it == m_map.end()) {
            throw std::out_of_range("Key not found in OrderedHashMap");
        }
        return it->second->second;
    }

    // Lookup
    bool contains(const Key& key) const {
        return m_map.find(key) != m_map.end();
    }

    Value* getptr(const Key& key) {
        auto it = m_map.find(key);
        return it != m_map.end() ? &it->second->second : nullptr;
    }

    const Value* getptr(const Key& key) const {
        auto it = m_map.find(key);
        return it != m_map.end() ? &it->second->second : nullptr;
    }

    // Removal
    bool erase(const Key& key) {
        auto it = m_map.find(key);
        if (it == m_map.end()) {
            return false;
        }
        m_list.erase(it->second);
        m_map.erase(it);
        return true;
    }

    void erase(ConstListIterator pos) {
        m_map.erase(pos->first);
        m_list.erase(pos);
    }

    void clear() {
        m_list.clear();
        m_map.clear();
    }

    // Iteration (preserves insertion order)
    ListIterator begin() { return m_list.begin(); }
    ListIterator end() { return m_list.end(); }
    ConstListIterator begin() const { return m_list.begin(); }
    ConstListIterator end() const { return m_list.end(); }
    ConstListIterator cbegin() const { return m_list.cbegin(); }
    ConstListIterator cend() const { return m_list.cend(); }

    // Reverse iteration
    auto rbegin() { return m_list.rbegin(); }
    auto rend() { return m_list.rend(); }
    auto rbegin() const { return m_list.rbegin(); }
    auto rend() const { return m_list.rend(); }

    // Front/back access (first/last inserted)
    Pair& front() { return m_list.front(); }
    const Pair& front() const { return m_list.front(); }
    Pair& back() { return m_list.back(); }
    const Pair& back() const { return m_list.back(); }

    // Move to back (useful for LRU caches)
    void move_to_back(const Key& key) {
        auto it = m_map.find(key);
        if (it != m_map.end()) {
            m_list.splice(m_list.end(), m_list, it->second);
        }
    }

    // Move to front
    void move_to_front(const Key& key) {
        auto it = m_map.find(key);
        if (it != m_map.end()) {
            m_list.splice(m_list.begin(), m_list, it->second);
        }
    }

    // Pop from front/back
    Pair pop_front() {
        if (m_list.empty()) {
            throw std::out_of_range("OrderedHashMap is empty");
        }
        Pair p = std::move(m_list.front());
        m_map.erase(p.first);
        m_list.pop_front();
        return p;
    }

    Pair pop_back() {
        if (m_list.empty()) {
            throw std::out_of_range("OrderedHashMap is empty");
        }
        Pair p = std::move(m_list.back());
        m_map.erase(p.first);
        m_list.pop_back();
        return p;
    }

    // Get keys in insertion order
    std::vector<Key> keys() const {
        std::vector<Key> result;
        result.reserve(size());
        for (const auto& p : m_list) {
            result.push_back(p.first);
        }
        return result;
    }

    // Get values in insertion order
    std::vector<Value> values() const {
        std::vector<Value> result;
        result.reserve(size());
        for (const auto& p : m_list) {
            result.push_back(p.second);
        }
        return result;
    }
};

// #############################################################################
// OrderedHashSet - Hash set that preserves insertion order
// #############################################################################
template <typename T, typename Hash = std::hash<T>, typename KeyEqual = std::equal_to<T>>
class OrderedHashSet {
private:
    std::list<T> m_list;
    std::unordered_map<T, typename std::list<T>::iterator, Hash, KeyEqual> m_map;

public:
    using Value = T;
    using Iterator = typename std::list<T>::iterator;
    using ConstIterator = typename std::list<T>::const_iterator;

    OrderedHashSet() = default;
    
    OrderedHashSet(std::initializer_list<T> init) {
        for (const auto& v : init) {
            insert(v);
        }
    }

    size_t size() const { return m_map.size(); }
    bool empty() const { return m_map.empty(); }

    void insert(const T& value) {
        if (!contains(value)) {
            m_list.push_back(value);
            m_map[value] = std::prev(m_list.end());
        }
    }

    void insert(T&& value) {
        if (!contains(value)) {
            m_list.push_back(std::move(value));
            m_map[value] = std::prev(m_list.end());
        }
    }

    bool contains(const T& value) const {
        return m_map.find(value) != m_map.end();
    }

    bool erase(const T& value) {
        auto it = m_map.find(value);
        if (it == m_map.end()) {
            return false;
        }
        m_list.erase(it->second);
        m_map.erase(it);
        return true;
    }

    void clear() {
        m_list.clear();
        m_map.clear();
    }

    Iterator begin() { return m_list.begin(); }
    Iterator end() { return m_list.end(); }
    ConstIterator begin() const { return m_list.begin(); }
    ConstIterator end() const { return m_list.end(); }
    ConstIterator cbegin() const { return m_list.cbegin(); }
    ConstIterator cend() const { return m_list.cend(); }

    auto rbegin() { return m_list.rbegin(); }
    auto rend() { return m_list.rend(); }

    const T& front() const { return m_list.front(); }
    const T& back() const { return m_list.back(); }

    std::vector<T> to_vector() const {
        return std::vector<T>(m_list.begin(), m_list.end());
    }
};

} // namespace godot

XTU_NAMESPACE_END

#endif // XTU_GODOT_XORDERED_HASH_MAP_HPP