// genesis/engine/entities/__init__.h

#pragma once

//------------------------------------------------------------------------------
// Genesis Engine - Entities Module
// Provides entity type enumeration and factory registration for all entity types.
//------------------------------------------------------------------------------

#include <memory>
#include <string>
#include <unordered_map>
#include <functional>
#include <vector>

namespace genesis {
namespace engine {

// Forward declare all entity types
class BaseEntity;
class RigidEntity;
class SoftEntity;
class ClothEntity;
class RopeEntity;
class FluidEntity;
class ParticleEntity;
class MPMParticleEntity;
class SPHParticleEntity;
class FEMEntity;
class PBDEntity;
class SFEntity;
class DroneEntity;
class EmitterEntity;
class HybridEntity;
class ToolEntity;

//------------------------------------------------------------------------------
// Entity type enumeration
//------------------------------------------------------------------------------
enum class EntityType : uint8_t {
    BASE = 0,
    RIGID = 1,
    SOFT = 2,
    CLOTH = 3,
    ROPE = 4,
    FLUID = 5,
    PARTICLE = 6,
    MPM = 7,
    SPH = 8,
    FEM = 9,
    PBD = 10,
    SF = 11,
    DRONE = 12,
    EMITTER = 13,
    HYBRID = 14,
    TOOL = 15,
    CUSTOM = 16
};

// Convert between enum and string
EntityType entity_type_from_string(const std::string& str);
std::string entity_type_to_string(EntityType type);

//------------------------------------------------------------------------------
// Entity creation function type
//------------------------------------------------------------------------------
using EntityCreator = std::function<std::shared_ptr<BaseEntity>()>;

//------------------------------------------------------------------------------
// Entity factory: registry for all entity types
//------------------------------------------------------------------------------
class EntityFactory {
public:
    static EntityFactory& instance();

    // Register an entity type with a creator function
    void register_entity(EntityType type, EntityCreator creator);
    void register_entity(const std::string& name, EntityCreator creator);

    // Create an entity instance by type or name
    std::shared_ptr<BaseEntity> create(EntityType type) const;
    std::shared_ptr<BaseEntity> create(const std::string& name) const;

    // Check if a type is registered
    bool is_registered(EntityType type) const;
    bool is_registered(const std::string& name) const;

    // Get list of registered names
    std::vector<std::string> registered_names() const;

    // Clear all registrations
    void clear();

private:
    EntityFactory() = default;
    std::unordered_map<EntityType, EntityCreator> creators_by_type_;
    std::unordered_map<std::string, EntityCreator> creators_by_name_;
};

//------------------------------------------------------------------------------
// Convenience creation functions for built-in types
//------------------------------------------------------------------------------
inline std::shared_ptr<RigidEntity> create_rigid_entity() {
    return std::dynamic_pointer_cast<RigidEntity>(EntityFactory::instance().create(EntityType::RIGID));
}

inline std::shared_ptr<SoftEntity> create_soft_entity() {
    return std::dynamic_pointer_cast<SoftEntity>(EntityFactory::instance().create(EntityType::SOFT));
}

inline std::shared_ptr<ClothEntity> create_cloth_entity() {
    return std::dynamic_pointer_cast<ClothEntity>(EntityFactory::instance().create(EntityType::CLOTH));
}

inline std::shared_ptr<RopeEntity> create_rope_entity() {
    return std::dynamic_pointer_cast<RopeEntity>(EntityFactory::instance().create(EntityType::ROPE));
}

inline std::shared_ptr<FluidEntity> create_fluid_entity() {
    return std::dynamic_pointer_cast<FluidEntity>(EntityFactory::instance().create(EntityType::FLUID));
}

inline std::shared_ptr<MPMEntity> create_mpm_entity() {
    return std::dynamic_pointer_cast<MPMEntity>(EntityFactory::instance().create(EntityType::MPM));
}

inline std::shared_ptr<SPHEntity> create_sph_entity() {
    return std::dynamic_pointer_cast<SPHEntity>(EntityFactory::instance().create(EntityType::SPH));
}

inline std::shared_ptr<FEMEntity> create_fem_entity() {
    return std::dynamic_pointer_cast<FEMEntity>(EntityFactory::instance().create(EntityType::FEM));
}

inline std::shared_ptr<PBDEntity> create_pbd_entity() {
    return std::dynamic_pointer_cast<PBDEntity>(EntityFactory::instance().create(EntityType::PBD));
}

inline std::shared_ptr<SFEntity> create_sf_entity() {
    return std::dynamic_pointer_cast<SFEntity>(EntityFactory::instance().create(EntityType::SF));
}

inline std::shared_ptr<DroneEntity> create_drone_entity() {
    return std::dynamic_pointer_cast<DroneEntity>(EntityFactory::instance().create(EntityType::DRONE));
}

inline std::shared_ptr<EmitterEntity> create_emitter_entity() {
    return std::dynamic_pointer_cast<EmitterEntity>(EntityFactory::instance().create(EntityType::EMITTER));
}

inline std::shared_ptr<HybridEntity> create_hybrid_entity() {
    return std::dynamic_pointer_cast<HybridEntity>(EntityFactory::instance().create(EntityType::HYBRID));
}

inline std::shared_ptr<ToolEntity> create_tool_entity() {
    return std::dynamic_pointer_cast<ToolEntity>(EntityFactory::instance().create(EntityType::TOOL));
}

//------------------------------------------------------------------------------
// Auto-registration helper macro
//------------------------------------------------------------------------------
struct EntityRegistrar {
    EntityRegistrar(EntityType type, const std::string& name, EntityCreator creator) {
        EntityFactory::instance().register_entity(type, creator);
        EntityFactory::instance().register_entity(name, creator);
    }
};

#define REGISTER_ENTITY(EntityClass, TypeEnum, TypeName) \
    static EntityRegistrar _registrar_##EntityClass( \
        TypeEnum, TypeName, \
        []() -> std::shared_ptr<BaseEntity> { return std::make_shared<EntityClass>(); })

} // namespace engine
} // namespace genesis