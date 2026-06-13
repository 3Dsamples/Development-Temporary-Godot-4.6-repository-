// genesis/engine/entities/base_entity.h

#pragma once

//------------------------------------------------------------------------------
// Base class for all simulation entities (rigid bodies, particles, cloth, etc.)
// Defines common interface for transform, physics properties, and queries.
//------------------------------------------------------------------------------

#include "genesis/datatypes.h"               // Vector3, Quat, Matrix4, AABB, etc.
#include "genesis/repr_base.h"               // Base class for string representation
#include <memory>                            // std::shared_ptr
#include <string>                            // std::string
#include <atomic>                            // std::atomic for thread-safe ID generation
#include <vector>                            // std::vector for collision shapes
#include <functional>                        // std::function for callbacks

namespace genesis {
namespace engine {

// Forward declaration of Scene (to avoid circular includes)
class Scene;

//------------------------------------------------------------------------------
// Collision shape types
//------------------------------------------------------------------------------
enum class CollisionShapeType : uint8_t {
    SPHERE = 0,                              // Sphere primitive
    BOX = 1,                                 // Box primitive
    CAPSULE = 2,                             // Capsule primitive
    CYLINDER = 3,                            // Cylinder primitive
    MESH = 4,                                // Triangle mesh (convex or concave)
    CONVEX_HULL = 5,                         // Convex hull from points
    PLANE = 6,                               // Infinite plane
    SDF = 7                                  // Signed distance field
};

//------------------------------------------------------------------------------
// Base collision shape structure (pure geometric description)
//------------------------------------------------------------------------------
struct CollisionShape {
    CollisionShapeType type;                 // Type of shape
    datatypes::Transformr local_transform;   // Transform relative to entity frame
    union {                                  // Shape-specific parameters (anonymous union)
        struct { double radius; } sphere;    // Sphere radius
        struct { datatypes::Vector3 half_extents; } box; // Box half-sizes
        struct { double radius; double height; } capsule; // Capsule parameters
        struct { double radius; double height; } cylinder; // Cylinder parameters
        struct { std::shared_ptr<class Mesh> mesh; } mesh; // Pointer to mesh data
        struct { std::vector<datatypes::Vector3> points; } convex; // Convex hull points
        struct { datatypes::Vector3 normal; double offset; } plane; // Plane definition
    } params;
    double margin = 0.0;                     // Collision margin (for GJK/EPA)
    uint32_t collision_group = 1;            // Bitmask for collision filtering
    uint32_t collision_mask = 0xFFFFFFFF;    // Bitmask for which groups to collide with
    
    // Constructor for sphere shape
    static CollisionShape make_sphere(double radius, const datatypes::Transformr& local = datatypes::Transformr());
    // Constructor for box shape
    static CollisionShape make_box(const datatypes::Vector3& half_extents, const datatypes::Transformr& local = datatypes::Transformr());
    // Constructor for capsule shape
    static CollisionShape make_capsule(double radius, double height, const datatypes::Transformr& local = datatypes::Transformr());
    // Constructor for cylinder shape
    static CollisionShape make_cylinder(double radius, double height, const datatypes::Transformr& local = datatypes::Transformr());
    // Constructor for mesh shape (from shared mesh resource)
    static CollisionShape make_mesh(std::shared_ptr<class Mesh> mesh, const datatypes::Transformr& local = datatypes::Transformr());
    // Constructor for convex hull from points
    static CollisionShape make_convex(const std::vector<datatypes::Vector3>& points, const datatypes::Transformr& local = datatypes::Transformr());
    // Constructor for plane shape
    static CollisionShape make_plane(const datatypes::Vector3& normal, double offset, const datatypes::Transformr& local = datatypes::Transformr());
};

//------------------------------------------------------------------------------
// BaseEntity abstract class - core interface for all simulation objects
//------------------------------------------------------------------------------
class BaseEntity : public ReprBase {
public:
    //----------------------------------------------------------------------
    // Construction and lifecycle
    //----------------------------------------------------------------------
    BaseEntity();                            // Default constructor (generates unique ID)
    explicit BaseEntity(const std::string& name); // Constructor with custom name
    virtual ~BaseEntity();                   // Virtual destructor for proper cleanup

    // Disable copy (entities are unique)
    BaseEntity(const BaseEntity&) = delete;
    BaseEntity& operator=(const BaseEntity&) = delete;

    // Enable move semantics
    BaseEntity(BaseEntity&& other) noexcept;
    BaseEntity& operator=(BaseEntity&& other) noexcept;

    //----------------------------------------------------------------------
    // Identification and type info
    //----------------------------------------------------------------------
    uint64_t id() const { return id_; }      // Get unique identifier
    const std::string& name() const { return name_; } // Get entity name
    void set_name(const std::string& name);  // Set entity name (must be unique in scene)
    virtual std::string type_name() const { return "BaseEntity"; } // Runtime type name

    //----------------------------------------------------------------------
    // Transform and motion state
    //----------------------------------------------------------------------
    const datatypes::Transformr& transform() const { return transform_; } // World transform
    void set_transform(const datatypes::Transformr& t); // Set world transform directly
    datatypes::Vector3 position() const;     // Get translation part
    void set_position(const datatypes::Vector3& pos); // Set translation
    datatypes::Quat rotation() const;        // Get rotation part
    void set_rotation(const datatypes::Quat& rot); // Set rotation
    void translate(const datatypes::Vector3& offset); // Translate by vector
    void rotate(const datatypes::Quat& delta); // Rotate by quaternion

    const datatypes::Vector3& velocity() const { return velocity_; } // Linear velocity
    void set_velocity(const datatypes::Vector3& vel); // Set linear velocity
    const datatypes::Vector3& angular_velocity() const { return angular_vel_; } // Angular velocity (axis-angle)
    void set_angular_velocity(const datatypes::Vector3& omega); // Set angular velocity
    void set_linear_damping(double damping) { linear_damping_ = damping; } // Set velocity damping factor
    void set_angular_damping(double damping) { angular_damping_ = damping; } // Set angular damping factor

    //----------------------------------------------------------------------
    // Physics properties
    //----------------------------------------------------------------------
    double mass() const { return mass_; }    // Total mass of entity
    void set_mass(double m);                 // Set mass (also updates inverse mass)
    double inverse_mass() const { return inv_mass_; } // 1/mass (0 for static)
    const datatypes::Vector3& inertia() const { return inertia_; } // Principal moments of inertia
    void set_inertia(const datatypes::Vector3& I); // Set diagonal inertia tensor
    datatypes::Vector3 inverse_inertia() const; // 1/I (component-wise)
    datatypes::Matrix3r world_inertia_tensor() const; // Inertia tensor in world frame
    datatypes::Vector3 local_center_of_mass() const { return local_com_; } // COM in local frame
    void set_local_center_of_mass(const datatypes::Vector3& com); // Set COM offset

    bool is_dynamic() const { return is_dynamic_; } // Can be moved by forces
    void set_dynamic(bool dynamic);          // Enable/disable dynamics
    bool is_enabled() const { return enabled_; } // Is entity active in simulation
    void set_enabled(bool enable);           // Enable/disable entity
    bool is_collidable() const { return collidable_; } // Can collide with others
    void set_collidable(bool collidable);    // Enable/disable collisions
    bool is_raycastable() const { return raycastable_; } // Can be hit by raycasts
    void set_raycastable(bool raycastable);  // Enable/disable raycasting

    //----------------------------------------------------------------------
    // Collision shapes
    //----------------------------------------------------------------------
    void add_collision_shape(const CollisionShape& shape); // Add a collision primitive
    void clear_collision_shapes();           // Remove all collision shapes
    const std::vector<CollisionShape>& collision_shapes() const { return shapes_; } // Get shapes
    datatypes::AABB world_aabb() const;      // Compute axis-aligned bounding box in world space
    datatypes::AABB local_aabb() const;      // Compute AABB in local entity space

    //----------------------------------------------------------------------
    // Forces and impulses
    //----------------------------------------------------------------------
    void apply_force(const datatypes::Vector3& force); // Add force at center of mass
    void apply_force(const datatypes::Vector3& force, const datatypes::Vector3& world_point); // Force at point
    void apply_torque(const datatypes::Vector3& torque); // Add torque (world space)
    void apply_impulse(const datatypes::Vector3& impulse); // Impulse at COM
    void apply_impulse(const datatypes::Vector3& impulse, const datatypes::Vector3& world_point); // Impulse at point
    void clear_forces();                     // Reset accumulated force/torque for next step
    const datatypes::Vector3& accumulated_force() const { return force_accum_; } // Get current net force
    const datatypes::Vector3& accumulated_torque() const { return torque_accum_; } // Get current net torque

    //----------------------------------------------------------------------
    // Integration (called by solvers)
    //----------------------------------------------------------------------
    virtual void integrate(double dt);        // Perform time integration (explicit Euler)
    virtual void integrate_velocity(double dt); // Update velocity from forces
    virtual void integrate_position(double dt); // Update position from velocity

    //----------------------------------------------------------------------
    // Queries
    //----------------------------------------------------------------------
    virtual datatypes::Vector3 velocity_at_point(const datatypes::Vector3& world_point) const; // Linear + rotational velocity at point
    virtual double inverse_mass(const datatypes::Vector3& world_point, const datatypes::Vector3& normal) const; // Effective inverse mass along normal
    virtual bool collide_with(const BaseEntity& other, double dt, std::vector<struct ContactPoint>& contacts) const; // Narrow-phase collision detection
    virtual struct RayHit ray_cast(const datatypes::Ray& ray) const; // Ray intersection test

    //----------------------------------------------------------------------
    // Scene linkage
    //----------------------------------------------------------------------
    Scene* scene() const { return scene_; }   // Get containing scene (nullptr if not in scene)
    void set_scene(Scene* scene);             // Called when added to/removed from scene

    //----------------------------------------------------------------------
    // Reset to initial state
    //----------------------------------------------------------------------
    virtual void reset();                     // Reset transform, velocity, and forces to initial values

    //----------------------------------------------------------------------
    // String representation (from ReprBase)
    //----------------------------------------------------------------------
    std::string repr() const override;        // Detailed string for debugging
    std::string str() const override;         // Human-readable string

protected:
    uint64_t id_;                            // Unique entity identifier (auto-generated)
    std::string name_;                       // Optional human-readable name
    Scene* scene_;                           // Weak reference to owning scene

    // Transform state
    datatypes::Transformr transform_;        // Current world transform
    datatypes::Transformr initial_transform_;// Transform at reset time
    datatypes::Vector3 velocity_;            // Linear velocity (world frame)
    datatypes::Vector3 angular_vel_;         // Angular velocity (world frame, axis-angle)
    datatypes::Vector3 initial_velocity_;    // Velocity at reset time
    datatypes::Vector3 initial_angular_vel_;// Angular velocity at reset time

    // Mass properties
    double mass_;                            // Total mass (kg)
    double inv_mass_;                        // Inverse mass (1/kg)
    datatypes::Vector3 inertia_;             // Principal moments of inertia (local frame)
    datatypes::Vector3 inv_inertia_;         // Inverse principal moments
    datatypes::Vector3 local_com_;           // Center of mass in local frame

    // Damping
    double linear_damping_;                  // Velocity damping coefficient (0-1)
    double angular_damping_;                 // Angular velocity damping coefficient (0-1)

    // Accumulated forces for current step
    datatypes::Vector3 force_accum_;         // Net force (world frame)
    datatypes::Vector3 torque_accum_;        // Net torque (world frame)

    // Flags
    bool is_dynamic_;                        // True if entity responds to forces
    bool enabled_;                           // True if entity is active
    bool collidable_;                        // True if entity participates in collisions
    bool raycastable_;                       // True if entity can be raycast

    // Collision geometry
    std::vector<CollisionShape> shapes_;     // List of collision primitives
    mutable datatypes::AABB cached_world_aabb_; // Cached world AABB (lazy evaluation)
    mutable bool aabb_dirty_;                // Flag indicating AABB needs recomputation

    // Static ID generator
    static std::atomic<uint64_t> next_id_;   // Atomic counter for unique IDs

    // Helper methods
    void update_cached_aabb() const;         // Recompute world AABB if dirty
    virtual void compute_world_aabb(datatypes::AABB& aabb) const; // Virtual for subclasses to override
};

//------------------------------------------------------------------------------
// RayHit structure (returned by ray_cast queries)
//------------------------------------------------------------------------------
struct RayHit {
    bool hit = false;                        // Whether an intersection occurred
    double t = std::numeric_limits<double>::max(); // Distance along ray to hit point
    datatypes::Vector3 point;                // World space hit position
    datatypes::Vector3 normal;               // Surface normal at hit point
    uint64_t entity_id = 0;                  // ID of hit entity
    std::shared_ptr<BaseEntity> entity;      // Shared pointer to hit entity
    uint32_t shape_index = 0;                // Index of hit collision shape
};

} // namespace engine
} // namespace genesis