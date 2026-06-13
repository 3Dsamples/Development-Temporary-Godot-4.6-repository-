// genesis/engine/material_property_manager.h
#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <functional>

namespace genesis {
namespace engine {

class BaseEntity;
class SoftTissueEntity;
class FEMEntity;
class SoftTissueSolver;
class RestorationController;
class JiggleMapper;
class FresnelMaterialModifier;

struct MaterialPropertySet {
    double stiffness   = 500.0;
    double damping     = 50.0;
    double restoration = 200.0;
    double jiggle      = 20.0;
    double hardness    = 0.5;
};

class MaterialPropertyManager {
public:
    MaterialPropertyManager();
    ~MaterialPropertyManager();

    void set_soft_tissue_solver(std::shared_ptr<SoftTissueSolver> solver);
    void set_restoration_controller(std::shared_ptr<RestorationController> ctrl);
    void set_jiggle_mapper(std::shared_ptr<JiggleMapper> mapper);
    void set_fresnel_modifier(std::shared_ptr<FresnelMaterialModifier> mod);

    void set_entity_properties(uint64_t entity_id, const MaterialPropertySet& props);
    MaterialPropertySet get_entity_properties(uint64_t entity_id) const;

    void set_global_properties(const MaterialPropertySet& props);
    MaterialPropertySet get_global_properties() const;

    void apply_properties();

    void set_change_callback(std::function<void(uint64_t, const MaterialPropertySet&)> callback);

    void set_stiffness(uint64_t entity_id, double value);
    void set_damping(uint64_t entity_id, double value);
    void set_restoration(uint64_t entity_id, double value);
    void set_jiggle(uint64_t entity_id, double value);
    void set_hardness(uint64_t entity_id, double value);

    // ** NEW: sync entity map (call once per frame with scene entities) **
    void sync_entities(const std::vector<std::shared_ptr<BaseEntity>>& entities);

private:
    MaterialPropertySet global_props_;
    std::unordered_map<uint64_t, MaterialPropertySet> entity_props_;
    std::unordered_map<uint64_t, std::weak_ptr<BaseEntity>> entity_map_; // NEW

    std::weak_ptr<SoftTissueSolver> solver_;
    std::weak_ptr<RestorationController> restoration_ctrl_;
    std::weak_ptr<JiggleMapper> jiggle_mapper_;
    std::weak_ptr<FresnelMaterialModifier> fresnel_modifier_;

    std::function<void(uint64_t, const MaterialPropertySet&)> change_callback_;

    void apply_to_entity(uint64_t id, const MaterialPropertySet& props);
    void apply_to_solver(uint64_t id, const MaterialPropertySet& props);
    void apply_to_restoration(uint64_t id, const MaterialPropertySet& props);
    void apply_to_jiggle(uint64_t id, const MaterialPropertySet& props);
    void apply_to_fresnel(uint64_t id, const MaterialPropertySet& props);
};

} // namespace engine
} // namespace genesis