// genesis/engine/entities/__init__.cpp

#include "genesis/engine/entities/__init__.h"
#include "genesis/engine/entities/rigid_entity.h"
#include "genesis/engine/entities/soft_entity.h"
#include "genesis/engine/entities/cloth_entity.h"
#include "genesis/engine/entities/rope_entity.h"
#include "genesis/engine/entities/fluid_entity.h"
#include "genesis/engine/entities/particle_entity.h"
#include "genesis/engine/entities/mpm_entity.h"
#include "genesis/engine/entities/sph_entity.h"
#include "genesis/engine/entities/fem_entity.h"
#include "genesis/engine/entities/pbd_entity.h"
#include "genesis/engine/entities/sf_entity.h"
#include "genesis/engine/entities/drone_entity.h"
#include "genesis/engine/entities/emitter.h"
#include "genesis/engine/entities/hybrid_entity.h"
#include "genesis/engine/entities/tool_entity.h"
#include <algorithm>
#include <cctype>

namespace genesis {
namespace engine {

//------------------------------------------------------------------------------
// String conversion for EntityType
//------------------------------------------------------------------------------
EntityType entity_type_from_string(const std::string& str) {
    // Convert input string to lowercase for case-insensitive comparison
    std::string lower = str;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    // Check against all known entity type strings
    if (lower == "rigid") return EntityType::RIGID;
    if (lower == "soft") return EntityType::SOFT;
    if (lower == "cloth") return EntityType::CLOTH;
    if (lower == "rope") return EntityType::ROPE;
    if (lower == "fluid") return EntityType::FLUID;
    if (lower == "particle") return EntityType::PARTICLE;
    if (lower == "mpm") return EntityType::MPM;
    if (lower == "sph") return EntityType::SPH;
    if (lower == "fem") return EntityType::FEM;
    if (lower == "pbd") return EntityType::PBD;
    if (lower == "sf") return EntityType::SF;
    if (lower == "drone") return EntityType::DRONE;
    if (lower == "emitter") return EntityType::EMITTER;
    if (lower == "hybrid") return EntityType::HYBRID;
    if (lower == "tool") return EntityType::TOOL;
    if (lower == "custom") return EntityType::CUSTOM;
    
    // Default to base type if no match found
    return EntityType::BASE;
}

std::string entity_type_to_string(EntityType type) {
    // Convert enum value to human-readable string
    switch (type) {
        case EntityType::RIGID:   return "RIGID";
        case EntityType::SOFT:    return "SOFT";
        case EntityType::CLOTH:   return "CLOTH";
        case EntityType::ROPE:    return "ROPE";
        case EntityType::FLUID:   return "FLUID";
        case EntityType::PARTICLE: return "PARTICLE";
        case EntityType::MPM:     return "MPM";
        case EntityType::SPH:     return "SPH";
        case EntityType::FEM:     return "FEM";
        case EntityType::PBD:     return "PBD";
        case EntityType::SF:      return "SF";
        case EntityType::DRONE:   return "DRONE";
        case EntityType::EMITTER: return "EMITTER";
        case EntityType::HYBRID:  return "HYBRID";
        case EntityType::TOOL:    return "TOOL";
        case EntityType::CUSTOM:  return "CUSTOM";
        default:                  return "BASE";
    }
}

//------------------------------------------------------------------------------
// EntityFactory implementation
//------------------------------------------------------------------------------
EntityFactory& EntityFactory::instance() {
    // Meyers' singleton pattern ensures thread-safe static initialization
    static EntityFactory factory;
    return factory;
}

void EntityFactory::register_entity(EntityType type, EntityCreator creator) {
    // Register a creator function associated with an entity type enum
    if (creator) {
        creators_by_type_[type] = creator;
    }
}

void EntityFactory::register_entity(const std::string& name, EntityCreator creator) {
    // Register a creator function associated with a string name
    if (creator && !name.empty()) {
        creators_by_name_[name] = creator;
    }
}

std::shared_ptr<BaseEntity> EntityFactory::create(EntityType type) const {
    // Look up the creator for the given type and invoke it
    auto it = creators_by_type_.find(type);
    if (it != creators_by_type_.end() && it->second) {
        return it->second();
    }
    // Return nullptr if type not registered
    return nullptr;
}

std::shared_ptr<BaseEntity> EntityFactory::create(const std::string& name) const {
    // First try exact case-sensitive name match
    auto it = creators_by_name_.find(name);
    if (it != creators_by_name_.end() && it->second) {
        return it->second();
    }
    
    // Then attempt case-insensitive match
    std::string lower_name = name;
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
    for (const auto& pair : creators_by_name_) {
        std::string lower_key = pair.first;
        std::transform(lower_key.begin(), lower_key.end(), lower_key.begin(), ::tolower);
        if (lower_key == lower_name && pair.second) {
            return pair.second();
        }
    }
    
    // Finally, try parsing name as EntityType enum string
    EntityType type = entity_type_from_string(name);
    return create(type);
}

bool EntityFactory::is_registered(EntityType type) const {
    // Check if a creator exists for the given enum type
    return creators_by_type_.find(type) != creators_by_type_.end();
}

bool EntityFactory::is_registered(const std::string& name) const {
    // Check case-sensitive match first
    if (creators_by_name_.find(name) != creators_by_name_.end()) {
        return true;
    }
    // Check case-insensitive match
    std::string lower_name = name;
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);
    for (const auto& pair : creators_by_name_) {
        std::string lower_key = pair.first;
        std::transform(lower_key.begin(), lower_key.end(), lower_key.begin(), ::tolower);
        if (lower_key == lower_name) {
            return true;
        }
    }
    return false;
}

std::vector<std::string> EntityFactory::registered_names() const {
    // Collect all registered string names into a vector
    std::vector<std::string> names;
    names.reserve(creators_by_name_.size());
    for (const auto& pair : creators_by_name_) {
        names.push_back(pair.first);
    }
    return names;
}

void EntityFactory::clear() {
    // Remove all registered creators
    creators_by_type_.clear();
    creators_by_name_.clear();
}

//------------------------------------------------------------------------------
// Auto-registration of built-in entity types
//------------------------------------------------------------------------------
// Register each concrete entity class using the REGISTER_ENTITY macro
// The registration occurs at static initialization time
REGISTER_ENTITY(RigidEntity, EntityType::RIGID, "Rigid");
REGISTER_ENTITY(SoftEntity, EntityType::SOFT, "Soft");
REGISTER_ENTITY(ClothEntity, EntityType::CLOTH, "Cloth");
REGISTER_ENTITY(RopeEntity, EntityType::ROPE, "Rope");
REGISTER_ENTITY(FluidEntity, EntityType::FLUID, "Fluid");
REGISTER_ENTITY(ParticleEntity, EntityType::PARTICLE, "Particle");
REGISTER_ENTITY(MPMEntity, EntityType::MPM, "MPM");
REGISTER_ENTITY(SPHEntity, EntityType::SPH, "SPH");
REGISTER_ENTITY(FEMEntity, EntityType::FEM, "FEM");
REGISTER_ENTITY(PBDEntity, EntityType::PBD, "PBD");
REGISTER_ENTITY(SFEntity, EntityType::SF, "SF");
REGISTER_ENTITY(DroneEntity, EntityType::DRONE, "Drone");
REGISTER_ENTITY(EmitterEntity, EntityType::EMITTER, "Emitter");
REGISTER_ENTITY(HybridEntity, EntityType::HYBRID, "Hybrid");
REGISTER_ENTITY(ToolEntity, EntityType::TOOL, "Tool");

} // namespace engine
} // namespace genesis