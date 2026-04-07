--- START OF FILE core/templates/rid.h ---

#ifndef RID_H
#define RID_H

#include "core/typedefs.h"
#include "src/big_int_core.h"

/**
 * RID
 * 
 * An opaque handle used to reference resources managed by Servers.
 * Uses BigIntCore to ensure unique identifiers even at galactic scales.
 * Aligned to 32 bytes for SIMD-safe processing in Warp parallel kernels.
 */
class ET_ALIGN_32 RID {
	BigIntCore _id;

public:
	_FORCE_INLINE_ bool is_valid() const { return !_id.is_zero(); }
	_FORCE_INLINE_ BigIntCore get_id() const { return _id; }

	_FORCE_INLINE_ bool operator==(const RID &p_rid) const { return _id == p_rid._id; }
	_FORCE_INLINE_ bool operator!=(const RID &p_rid) const { return _id != p_rid._id; }
	_FORCE_INLINE_ bool operator<(const RID &p_rid) const { return _id < p_rid._id; }
	_FORCE_INLINE_ bool operator<=(const RID &p_rid) const { return _id <= p_rid._id; }
	_FORCE_INLINE_ bool operator>(const RID &p_rid) const { return _id > p_rid._id; }
	_FORCE_INLINE_ bool operator>=(const RID &p_rid) const { return _id >= p_rid._id; }

	_FORCE_INLINE_ uint32_t hash() const { return _id.hash(); }

	_FORCE_INLINE_ RID() : _id(0LL) {}
	_FORCE_INLINE_ explicit RID(const BigIntCore &p_id) : _id(p_id) {}
};

/**
 * RID_Owner_Base
 * 
 * Base class for managing the lifecycle of resources associated with RIDs.
 * Designed to interface with the EnTT Registry for SoA component storage.
 */
class RID_Owner_Base {
public:
	virtual void free(RID p_rid) = 0;
	virtual bool owns(RID p_rid) const = 0;
	virtual ~RID_Owner_Base() {}
};

#endif // RID_H

--- END OF FILE core/templates/rid.h ---
