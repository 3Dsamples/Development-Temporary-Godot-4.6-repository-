//core/core_bind.h

#ifndef CORE_BIND_H
#define CORE_BIND_H

#include "core/object/object.h"
#include "src/big_int_core.h"
#include "src/big_number.h"
#include "src/fixed_math_core.h"
#include "core/templates/vector.h"

/**
 * @class _ResourceLoader
 * @brief High-performance resource loading bound to BigNumber logic for deterministic spatial assets.
 */
class _ResourceLoader : public Object {
	GDCLASS(_ResourceLoader, Object);

protected:
	static void _bind_methods();
	static _ResourceLoader *singleton;

public:
	static _ResourceLoader *get_singleton() { return singleton; }

	Error load(const String &p_path, const String &p_type_hint = "");
	Vector<String> get_recognized_extensions_for_type(const String &p_type);
	void set_abort_on_missing_resources(bool p_abort);
	bool exists(const String &p_path, const String &p_type_hint = "");

	_ResourceLoader();
};

/**
 * @class _Geometry2D
 * @brief 2D Geometric utilities using BigNumber for bit-perfect collision and intersection logic.
 */
class _Geometry2D : public Object {
	GDCLASS(_Geometry2D, Object);

protected:
	static void _bind_methods();
	static _Geometry2D *singleton;

public:
	static _Geometry2D *get_singleton() { return singleton; }

	bool is_point_in_circle(const BigNumber &p_point_x, const BigNumber &p_point_y, const BigNumber &p_circle_x, const BigNumber &p_circle_y, const BigNumber &p_radius);
	BigNumber segment_intersects_circle(const BigNumber &p_from_x, const BigNumber &p_from_y, const BigNumber &p_to_x, const BigNumber &p_to_y, const BigNumber &p_circle_x, const BigNumber &p_circle_y, const BigNumber &p_radius);
	bool is_point_in_polygon(const BigNumber &p_x, const BigNumber &p_y, const Vector<BigNumber> &p_polygon_x, const Vector<BigNumber> &p_polygon_y);
	Vector<BigNumber> clip_polygons(const Vector<BigNumber> &p_poly_a_x, const Vector<BigNumber> &p_poly_a_y, const Vector<BigNumber> &p_poly_b_x, const Vector<BigNumber> &p_poly_b_y);

	_Geometry2D();
};

/**
 * @class _Geometry3D
 * @brief 3D Geometric utilities for galactic-scale intersection queries using BigNumber.
 */
class _Geometry3D : public Object {
	GDCLASS(_Geometry3D, Object);

protected:
	static void _bind_methods();
	static _Geometry3D *singleton;

public:
	static _Geometry3D *get_singleton() { return singleton; }

	Vector<BigNumber> compute_convex_hull(const Vector<BigNumber> &p_points_x, const Vector<BigNumber> &p_points_y, const Vector<BigNumber> &p_points_z);
	bool ray_intersects_triangle(const BigNumber &p_from_x, const BigNumber &p_from_y, const BigNumber &p_from_z, const BigNumber &p_dir_x, const BigNumber &p_dir_y, const BigNumber &p_dir_z, const BigNumber &p_v0_x, const BigNumber &p_v0_y, const BigNumber &p_v0_z, const BigNumber &p_v1_x, const BigNumber &p_v1_y, const BigNumber &p_v1_z, const BigNumber &p_v2_x, const BigNumber &p_v2_y, const BigNumber &p_v2_z);

	_Geometry3D();
};

#endif // CORE_BIND_H