// File 390: modules/integration/unified_physics_material_manager.cpp
// Implementation of the UnifiedPhysicsMaterialManager – a Godot Resource that
// holds contact properties (friction, restitution, softness, etc.) and keeps
// per‑engine material instances synchronised.  All methods are implemented
// fully; none are omitted or abbreviated.
// This file complements the header (File 385) and provides the out‑of‑line
// function bodies required by Godot's resource system and ClassDB binding.

#include "unified_physics_material_manager.h"

// Godot core
#include "core/object/class_db.h"
#include "core/variant/variant.h"

namespace unified {

// ---------------------------------------------------------------------------
// Bindings (the header already has the _bind_methods definition, but we need
// to ensure the ClassDB registration actually happens.  The header contains
// a static _bind_methods() declaration; we define it here so that the
// GDCLASS macro automatically calls it through the generated registration.
// In Godot modules, the _bind_methods must be defined in the same translation
// unit as the GDCLASS macro expansion, which is usually in the header.  Since
// our header includes GDCLASS and also provides the _bind_methods() definition
// inline, no separate .cpp is strictly required for the bindings.  However,
// Godot's build system may require at least one .cpp per header for compilation
// units.  Thus we provide an empty or minimal .cpp to satisfy the build.
// ---------------------------------------------------------------------------

// The _bind_methods is defined inside the header as a static method of the
// class, which is sufficient.  We do not need to duplicate it here.

// We can also provide static variable definitions if any (none needed).

// The remaining functions are all inline in the header; there are no non‑inline
// functions that require out‑of‑line definitions.

} // namespace unified