// genesis/engine/entities/sf_entity.h

#pragma once

//------------------------------------------------------------------------------
// SF Entity class - Signed Distance Field fluid entity.
// Represents a fluid volume defined by a level set function on a grid.
// Can optionally contain particles for hybrid particle-level set methods.
//------------------------------------------------------------------------------

#include "genesis/engine/entities/base_entity.h"   // Base class: BaseEntity
#include "genesis/datatypes.h"                      // Vector3, AABB, real, etc.
#include <vector>                                   // std::vector for particles and grid data
#include <memory>                                   // std::shared_ptr
#include <string>                                   // std::string

namespace genesis {
namespace engine {

// Forward declaration
class Mesh;

//------------------------------------------------------------------------------
// SF Configuration
//------------------------------------------------------------------------------
struct SFConfig {
    // Fluid properties
    double density = 1000.0;                        // Rest density (kg/m³)
    double viscosity = 0.01;                        // Dynamic viscosity (Pa·s)
    double surface_tension = 0.072;                 // Surface tension coefficient (N/m)
    
    // Level set parameters
    double reinitialization_speed = 1.0;            // Speed for reinitialization (CFL-like)
    int reinitialization_steps = 3;                 // Number of fast marching steps per frame
    
    // Particle sampling (for hybrid particle-level set)
    bool use_particles = false;                     // Enable particle correction
    int particles_per_cell = 8;                     // Number of particles per grid cell
    double particle_radius = 0.01;                  // Radius of correction particles
    
    // Boundary conditions
    bool use_open_boundary = false;                 // Allow fluid to exit domain
    double pressure_extrapolation = 0.0;            // External pressure value
};

//------------------------------------------------------------------------------
// SF Entity class - Signed Distance Field fluid volume
//------------------------------------------------------------------------------
class SFEntity : public BaseEntity {
public:
    //----------------------------------------------------------------------
    // Construction
    //----------------------------------------------------------------------
    SFEntity();                                     // Default constructor
    explicit SFEntity(const std::string& name);     // Named constructor
    virtual ~SFEntity();                            // Destructor

    //----------------------------------------------------------------------
    // Configuration
    //----------------------------------------------------------------------
    void set_sf_config(const SFConfig& config);     // Set fluid configuration
    const SFConfig& sf_config() const { return sf_config_; }

    //----------------------------------------------------------------------
    // Level set grid access (for solver)
    //----------------------------------------------------------------------
    // Set the level set grid dimensions and origin
    void set_grid(int nx, int ny, int nz, const datatypes::Vector3& origin, double cell_size);
    
    // Grid dimensions
    int grid_nx() const { return nx_; }
    int grid_ny() const { return ny_; }
    int grid_nz() const { return nz_; }
    datatypes::Vector3 grid_origin() const { return origin_; }
    double cell_size() const { return dx_; }
    
    // Level set values (signed distance)
    const std::vector<double>& level_set() const { return phi_; }
    std::vector<double>& level_set() { return phi_; }
    void set_level_set(const std::vector<double>& phi);
    double level_set(int i, int j, int k) const;
    void set_level_set(int i, int j, int k, double value);
    
    // Velocity field (staggered MAC grid)
    const std::vector<double>& velocity_u() const { return u_; }
    const std::vector<double>& velocity_v() const { return v_; }
    const std::vector<double>& velocity_w() const { return w_; }
    std::vector<double>& velocity_u() { return u_; }
    std::vector<double>& velocity_v() { return v_; }
    std::vector<double>& velocity_w() { return w_; }
    
    // Pressure field (cell-centered)
    const std::vector<double>& pressure() const { return pressure_; }
    std::vector<double>& pressure() { return pressure_; }
    
    // Cell flags (fluid, solid, empty)
    const std::vector<uint8_t>& cell_flags() const { return flags_; }
    std::vector<uint8_t>& cell_flags() { return flags_; }
    
    //----------------------------------------------------------------------
    // Initialization from geometry
    //----------------------------------------------------------------------
    // Initialize level set from a mesh (signed distance computation)
    void init_from_mesh(const Mesh& mesh);
    
    // Initialize from a bounding box (fill interior)
    void init_from_box(const datatypes::AABB& box);
    
    // Initialize from a sphere
    void init_from_sphere(const datatypes::Vector3& center, double radius);
    
    // Add a solid obstacle (modifies cell flags and level set)
    void add_solid_obstacle(const Mesh& mesh);
    void add_solid_box(const datatypes::AABB& box);
    void clear_solids();
    
    //----------------------------------------------------------------------
    // Particle management (for hybrid particle-level set)
    //----------------------------------------------------------------------
    bool has_particles() const { return sf_config_.use_particles && !particle_positions_.empty(); }
    size_t particle_count() const { return particle_positions_.size(); }
    
    const std::vector<datatypes::Vector3>& particle_positions() const { return particle_positions_; }
    std::vector<datatypes::Vector3>& particle_positions() { return particle_positions_; }
    
    const std::vector<datatypes::Vector3>& particle_velocities() const { return particle_velocities_; }
    std::vector<datatypes::Vector3>& particle_velocities() { return particle_velocities_; }
    
    const std::vector<double>& particle_masses() const { return particle_masses_; }
    std::vector<double>& particle_masses() { return particle_masses_; }
    
    double particle_radius() const { return sf_config_.particle_radius; }
    uint32_t material_id() const { return material_id_; }
    void set_material_id(uint32_t id) { material_id_ = id; }
    
    // Add/remove particles
    void add_particles(const std::vector<datatypes::Vector3>& positions,
                       const std::vector<datatypes::Vector3>& velocities,
                       const std::vector<double>& masses);
    void remove_particles(const std::vector<bool>& mask);
    void clear_particles();
    
    // Seed particles inside the fluid volume
    void seed_particles();
    
    // Set particle states (used by solver)
    void set_particle_states(const std::vector<datatypes::Vector3>& positions,
                             const std::vector<datatypes::Vector3>& velocities);
    
    //----------------------------------------------------------------------
    // Overrides from BaseEntity
    //----------------------------------------------------------------------
    virtual void integrate(double dt) override;     // SF integration is solver-driven
    virtual void reset() override;
    virtual datatypes::AABB world_aabb() const override;
    
    //----------------------------------------------------------------------
    // String representation
    //----------------------------------------------------------------------
    virtual std::string type_name() const override { return "SFEntity"; }
    virtual std::string repr() const override;
    virtual std::string str() const override;

private:
    SFConfig sf_config_;                            // Fluid configuration
    
    // Grid definition
    int nx_ = 0, ny_ = 0, nz_ = 0;                  // Grid dimensions
    datatypes::Vector3 origin_;                     // Grid origin (minimum corner)
    double dx_ = 0.05;                              // Cell size
    
    // Grid data
    std::vector<double> phi_;                       // Level set (signed distance)
    std::vector<double> u_, v_, w_;                 // Staggered velocity components
    std::vector<double> pressure_;                  // Cell-centered pressure
    std::vector<uint8_t> flags_;                    // Cell flags (0=empty,1=fluid,2=solid)
    
    // Particle data (for hybrid method)
    std::vector<datatypes::Vector3> particle_positions_;
    std::vector<datatypes::Vector3> particle_velocities_;
    std::vector<double> particle_masses_;
    uint32_t material_id_ = 0;
    
    // Helper methods
    int cell_index(int i, int j, int k) const { return i + nx_ * (j + ny_ * k); }
    int u_index(int i, int j, int k) const { return i + (nx_+1) * (j + ny_ * k); }
    int v_index(int i, int j, int k) const { return i + nx_ * (j + (ny_+1) * k); }
    int w_index(int i, int j, int k) const { return i + nx_ * (j + ny_ * (k+1)); }
    
    bool valid_cell(int i, int j, int k) const {
        return i >= 0 && i < nx_ && j >= 0 && j < ny_ && k >= 0 && k < nz_;
    }
    
    void resize_grid();                             // Allocate arrays based on nx_, ny_, nz_
    void compute_sdf_from_mesh(const Mesh& mesh);   // Brute force SDF computation
    void reinitialize_level_set();                  // Fast marching to maintain signed distance property
};

} // namespace engine
} // namespace genesis