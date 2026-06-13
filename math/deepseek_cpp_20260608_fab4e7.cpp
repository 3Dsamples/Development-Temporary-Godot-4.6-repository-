// File 384: modules/integration/unified_collision_filter.h
// Cross‑Engine Collision Filter – unified collision layer / mask system
// that operates across all physics engines (Newton, Genesis, Vienna, Wicked).
// Each body is assigned a 64‑bit layer mask and a 64‑bit collision mask.
// The filter is evaluated before narrow‑phase collision detection and
// can also be used to disqualify pairs during broad‑phase querying.
// All hot‑path lookups use flat arrays or open‑addressing hash maps.

#ifndef INTEGRATION_UNIFIED_COLLISION_FILTER_H
#define INTEGRATION_UNIFIED_COLLISION_FILTER_H

#include "core/object/ref_counted.h"
#include "core/templates/hash_map.h"
#include "core/templates/local_vector.h"
#include "core/typedefs.h"

namespace unified {

class UnifiedCollisionFilter : public RefCounted {
    GDCLASS(UnifiedCollisionFilter, RefCounted);

public:
    // Layer and mask definitions (0 = no collision, 0xFFFFFFFFFFFFFFFF = all).
    static constexpr uint64_t LAYER_ALL = ~0ULL;

    struct BodyFilterInfo {
        uint64_t layer = 1;                     // which layers this body belongs to
        uint64_t mask  = LAYER_ALL;             // which layers it can collide with
    };

private:
    // Storage: maps body IDs (engine‑specific) to their filter info.
    // We use a separate map per engine to avoid key collisions.
    struct EngineFilterMap {
        HashMap<uint64_t, BodyFilterInfo> body_filters;
        HashMap<uint64_t, BodyFilterInfo> static_body_filters; // static bodies have separate lookup? Not needed; we'll just use body_filters.
    };

    EngineFilterMap engine_maps[4];  // 0=Newton, 1=Genesis, 2=Vienna, 3=Wicked

    // Additionally, we keep a list of explicit disabled pairs (engine‑agnostic? We need to handle pairs between different engines separately.)
    // A disabled pair is identified by (engine_a, body_a, engine_b, body_b).
    struct DisabledPairKey {
        uint8_t eng_a, eng_b;
        uint64_t id_a, id_b;
        DisabledPairKey() : eng_a(0), eng_b(0), id_a(0), id_b(0) {}
        DisabledPairKey(uint8_t ea, uint64_t ia, uint8_t eb, uint64_t ib)
            : eng_a(ea), id_a(ia), eng_b(eb), id_b(ib) {
            if (eng_a > eng_b || (eng_a == eng_b && id_a > id_b)) {
                SWAP(eng_a, eng_b);
                SWAP(id_a, id_b);
            }
        }
        bool operator==(const DisabledPairKey &o) const {
            return eng_a == o.eng_a && eng_b == o.eng_b && id_a == o.id_a && id_b == o.id_b;
        }
        struct Hash {
            uint64_t operator()(const DisabledPairKey &k) const {
                uint64_t h = (uint64_t(k.eng_a) << 56) | (uint64_t(k.eng_b) << 48) |
                             (k.id_a & 0xFFFFFF) | ((k.id_b & 0xFFFFFF) << 24);
                // Murmur3 mix
                h ^= h >> 33;
                h *= 0xff51afd7ed558ccdULL;
                h ^= h >> 33;
                h *= 0xc4ceb9fe1a85ec53ULL;
                h ^= h >> 33;
                return h;
            }
        };
    };

    HashSet<DisabledPairKey, DisabledPairKey::Hash> disabled_pairs;

public:
    UnifiedCollisionFilter() {}

    // Engine index constants
    static constexpr int ENGINE_NEWTON  = 0;
    static constexpr int ENGINE_GENESIS = 1;
    static constexpr int ENGINE_VIENNA  = 2;
    static constexpr int ENGINE_WICKED  = 3;

    // Set the filter info for a body.
    void set_body_filter(int p_engine, uint64_t p_body_id, const BodyFilterInfo &p_info) {
        ERR_FAIL_INDEX(p_engine, 4);
        engine_maps[p_engine].body_filters[p_body_id] = p_info;
    }

    // Get the filter info for a body.
    BodyFilterInfo get_body_filter(int p_engine, uint64_t p_body_id) const {
        ERR_FAIL_INDEX_V(p_engine, 4, BodyFilterInfo{1, LAYER_ALL});
        HashMap<uint64_t, BodyFilterInfo>::ConstIterator it =
            engine_maps[p_engine].body_filters.find(p_body_id);
        if (it) return it->value;
        return BodyFilterInfo{1, LAYER_ALL};
    }

    // Remove a body's filter info.
    void remove_body_filter(int p_engine, uint64_t p_body_id) {
        ERR_FAIL_INDEX(p_engine, 4);
        engine_maps[p_engine].body_filters.erase(p_body_id);
    }

    // Disable collision between two bodies (any engines).
    void disable_pair(int p_eng_a, uint64_t p_id_a, int p_eng_b, uint64_t p_id_b) {
        disabled_pairs.insert(DisabledPairKey(p_eng_a, p_id_a, p_eng_b, p_id_b));
    }

    // Re‑enable collision between two bodies.
    void enable_pair(int p_eng_a, uint64_t p_id_a, int p_eng_b, uint64_t p_id_b) {
        disabled_pairs.erase(DisabledPairKey(p_eng_a, p_id_a, p_eng_b, p_id_b));
    }

    // Check if a pair is explicitly disabled.
    bool is_pair_disabled(int p_eng_a, uint64_t p_id_a, int p_eng_b, uint64_t p_id_b) const {
        return disabled_pairs.has(DisabledPairKey(p_eng_a, p_id_a, p_eng_b, p_id_b));
    }

    // Main filter call: returns true if the two bodies should collide.
    // Considers layer/mask and explicit pair disabling.
    bool can_collide(int p_eng_a, uint64_t p_id_a, int p_eng_b, uint64_t p_id_b) const {
        if (is_pair_disabled(p_eng_a, p_id_a, p_eng_b, p_id_b))
            return false;

        BodyFilterInfo info_a = get_body_filter(p_eng_a, p_id_a);
        BodyFilterInfo info_b = get_body_filter(p_eng_b, p_id_b);

        // Standard collision filter: (layer_a & mask_b) != 0 && (layer_b & mask_a) != 0
        return (info_a.layer & info_b.mask) && (info_b.layer & info_a.mask);
    }
};

} // namespace unified

#endif // INTEGRATION_UNIFIED_COLLISION_FILTER_H