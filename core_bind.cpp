//core/core_bind.cpp

#include "core/core_bind.h"
#include "core/io/resource_loader.h"
#include "core/object/class_db.h"
#include "src/big_number.h"
#include "src/fixed_math_core.h"

// --- _ResourceLoader ---

_ResourceLoader *_ResourceLoader::singleton = nullptr;

void _ResourceLoader::_bind_methods() {
	ClassDB::bind_method(D_METHOD("load", "path", "type_hint"), &_ResourceLoader::load, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("get_recognized_extensions_for_type", "type"), &_ResourceLoader::get_recognized_extensions_for_type);
	ClassDB::bind_method(D_METHOD("set_abort_on_missing_resources", "abort"), &_ResourceLoader::set_abort_on_missing_resources);
	ClassDB::bind_method(D_METHOD("exists", "path", "type_hint"), &_ResourceLoader::exists, DEFVAL(""));
}

Error _ResourceLoader::load(const String &p_path, const String &p_type_hint) {
	return ResourceLoader::load_internal(p_path, p_type_hint);
}

Vector<String> _ResourceLoader::get_recognized_extensions_for_type(const String &p_type) {
	return ResourceLoader::get_recognized_extensions_for_type_internal(p_type);
}

void _ResourceLoader::set_abort_on_missing_resources(bool p_abort) {
	ResourceLoader::set_abort_on_missing_resources_internal(p_abort);
}

bool _ResourceLoader::exists(const String &p_path, const String &p_type_hint) {
	return ResourceLoader::exists_internal(p_path, p_type_hint);
}

_ResourceLoader::_ResourceLoader() {
	singleton = this;
}

// --- _Geometry2D ---

_Geometry2D *_Geometry2D::singleton = nullptr;

void _Geometry2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("is_point_in_circle", "point_x", "point_y", "circle_x", "circle_y", "radius"), &_Geometry2D::is_point_in_circle);
	ClassDB::bind_method(D_METHOD("segment_intersects_circle", "from_x", "from_y", "to_x", "to_y", "circle_x", "circle_y", "radius"), &_Geometry2D::segment_intersects_circle);
	ClassDB::bind_method(D_METHOD("is_point_in_polygon", "x", "y", "polygon_x", "polygon_y"), &_Geometry2D::is_point_in_polygon);
}

bool _Geometry2D::is_point_in_circle(const BigNumber &p_point_x, const BigNumber &p_point_y, const BigNumber &p_circle_x, const BigNumber &p_circle_y, const BigNumber &p_radius) {
	BigNumber dx = p_point_x - p_circle_x;
	BigNumber dy = p_point_y - p_circle_y;
	BigNumber dist_sq = (dx * dx) + (dy * dy);
	BigNumber rad_sq = p_radius * p_radius;
	return dist_sq <= rad_sq;
}

BigNumber _Geometry2D::segment_intersects_circle(const BigNumber &p_from_x, const BigNumber &p_from_y, const BigNumber &p_to_x, const BigNumber &p_to_y, const BigNumber &p_circle_x, const BigNumber &p_circle_y, const BigNumber &p_radius) {
	// Vector logic using BigNumber
	BigNumber dx = p_to_x - p_from_x;
	BigNumber dy = p_to_y - p_from_y;
	BigNumber fx = p_from_x - p_circle_x;
	BigNumber fy = p_from_y - p_circle_y;

	BigNumber a = (dx * dx) + (dy * dy);
	BigNumber b = BigNumber(2) * ((fx * dx) + (fy * dy));
	BigNumber c = ((fx * fx) + (fy * fy)) - (p_radius * p_radius);

	BigNumber discriminant = (b * b) - (BigNumber(4) * a * c);
	if (discriminant < BigNumber(0)) {
		return BigNumber(-1); // No intersection
	}

	// Quadratic formula: t = (-b - sqrt(D)) / (2a)
	FixedMathCore::fixed_t d_fixed = FixedMathCore::from_double(discriminant.to_double());
	FixedMathCore::fixed_t sqrt_d_fixed = FixedMathCore::sqrt(d_fixed);
	BigNumber sqrt_d = BigNumber(BigIntCore(0), sqrt_d_fixed);

	BigNumber t = (-b - sqrt_d) / (BigNumber(2) * a);
	if (t >= BigNumber(0) && t <= BigNumber(1)) {
		return t;
	}

	return BigNumber(-1);
}

bool _Geometry2D::is_point_in_polygon(const BigNumber &p_x, const BigNumber &p_y, const Vector<BigNumber> &p_polygon_x, const Vector<BigNumber> &p_polygon_y) {
	if (p_polygon_x.size() < 3) return false;
	bool inside = false;
	int j = p_polygon_x.size() - 1;
	for (int i = 0; i < p_polygon_x.size(); i++) {
		if (((p_polygon_y[i] > p_y) != (p_polygon_y[j] > p_y)) &&
				(p_x < (p_polygon_x[j] - p_polygon_x[i]) * (p_y - p_polygon_y[i]) / (p_polygon_y[j] - p_polygon_y[i]) + p_polygon_x[i])) {
			inside = !inside;
		}
		j = i;
	}
	return inside;
}

_Geometry2D::_Geometry2D() {
	singleton = this;
}

// --- _Geometry3D ---

_Geometry3D *_Geometry3D::singleton = nullptr;

void _Geometry3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("ray_intersects_triangle", "from_x", "from_y", "from_z", "dir_x", "dir_y", "dir_z", "v0_x", "v0_y", "v0_z", "v1_x", "v1_y", "v1_z", "v2_x", "v2_y", "v2_z"), &_Geometry3D::ray_intersects_triangle);
}

bool _Geometry3D::ray_intersects_triangle(const BigNumber &p_from_x, const BigNumber &p_from_y, const BigNumber &p_from_z, const BigNumber &p_dir_x, const BigNumber &p_dir_y, const BigNumber &p_dir_z, const BigNumber &p_v0_x, const BigNumber &p_v0_y, const BigNumber &p_v0_z, const BigNumber &p_v1_x, const BigNumber &p_v1_y, const BigNumber &p_v1_z, const BigNumber &p_v2_x, const BigNumber &p_v2_y, const BigNumber &p_v2_z) {
	// Möller–Trumbore intersection algorithm with BigNumber
	BigNumber edge1_x = p_v1_x - p_v0_x;
	BigNumber edge1_y = p_v1_y - p_v0_y;
	BigNumber edge1_z = p_v1_z - p_v0_z;
	
	BigNumber edge2_x = p_v2_x - p_v0_x;
	BigNumber edge2_y = p_v2_y - p_v0_y;
	BigNumber edge2_z = p_v2_z - p_v0_z;

	// Cross product: dir x edge2
	BigNumber h_x = (p_dir_y * edge2_z) - (p_dir_z * edge2_y);
	BigNumber h_y = (p_dir_z * edge2_x) - (p_dir_x * edge2_z);
	BigNumber h_z = (p_dir_x * edge2_y) - (p_dir_y * edge2_x);

	// Dot product: edge1 . h
	BigNumber a = (edge1_x * h_x) + (edge1_y * h_y) + (edge1_z * h_z);

	if (a > BigNumber("-0.0000001") && a < BigNumber("0.0000001")) {
		return false; // Ray is parallel
	}

	BigNumber f = BigNumber(1) / a;
	BigNumber s_x = p_from_x - p_v0_x;
	BigNumber s_y = p_from_y - p_v0_y;
	BigNumber s_z = p_from_z - p_v0_z;

	// u = f * (s . h)
	BigNumber u = f * ((s_x * h_x) + (s_y * h_y) + (s_z * h_z));
	if (u < BigNumber(0) || u > BigNumber(1)) {
		return false;
	}

	// q = s x edge1
	BigNumber q_x = (s_y * edge1_z) - (s_z * edge1_y);
	BigNumber q_y = (s_z * edge1_x) - (s_x * edge1_z);
	BigNumber q_z = (s_x * edge1_y) - (s_y * edge1_x);

	// v = f * (dir . q)
	BigNumber v = f * ((p_dir_x * q_x) + (p_dir_y * q_y) + (p_dir_z * q_z));
	if (v < BigNumber(0) || u + v > BigNumber(1)) {
		return false;
	}

	// t = f * (edge2 . q)
	BigNumber t = f * ((edge2_x * q_x) + (edge2_y * q_y) + (edge2_z * q_z));
	return t > BigNumber("0.0000001");
}

_Geometry3D::_Geometry3D() {
	singleton = this;
}