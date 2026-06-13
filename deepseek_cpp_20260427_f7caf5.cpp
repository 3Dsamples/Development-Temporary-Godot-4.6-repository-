// File 218: modules/newton/src/collision/newton_collision.cpp
// Bindings and constructors for base collision shapes (sphere, box, capsule).

#include "newton_collision.h"

namespace newton {

// --- NewtonCollisionSphere bindings ---
void NewtonCollisionSphere::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_radius", "radius"), &NewtonCollisionSphere::set_radius);
	ClassDB::bind_method(D_METHOD("get_radius"), &NewtonCollisionSphere::get_radius);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius"), "set_radius", "get_radius");
}

// --- NewtonCollisionBox bindings ---
void NewtonCollisionBox::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_half_extents", "extents"), &NewtonCollisionBox::set_half_extents);
	ClassDB::bind_method(D_METHOD("get_half_extents"), &NewtonCollisionBox::get_half_extents);
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "half_extents"), "set_half_extents", "get_half_extents");
}

// --- NewtonCollisionCapsule bindings ---
void NewtonCollisionCapsule::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_radius", "r"), &NewtonCollisionCapsule::set_radius);
	ClassDB::bind_method(D_METHOD("get_radius"), &NewtonCollisionCapsule::get_radius);
	ClassDB::bind_method(D_METHOD("set_height", "h"), &NewtonCollisionCapsule::set_height);
	ClassDB::bind_method(D_METHOD("get_height"), &NewtonCollisionCapsule::get_height);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "radius"), "set_radius", "get_radius");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "height"), "set_height", "get_height");
}

} // namespace newton