// genesis/engine/entities/base_entity.cpp

#include "genesis/engine/entities/base_entity.h" // Include corresponding header
#include "genesis/engine/scene.h"                // Scene reference needed for world queries
#include "genesis/engine/mesh.h"                 // For mesh collision shape support
#include <sstream>                               // std::ostringstream for repr
#include <iomanip>                               // std::setw, std::setfill
#include <cmath>                                 // std::abs, std::sqrt

namespace genesis {
namespace engine {

//------------------------------------------------------------------------------
// Static ID generator initialization
//------------------------------------------------------------------------------
std::atomic<uint64_t> BaseEntity::next_id_{1}; // Start IDs from 1 (0 reserved for invalid)

//------------------------------------------------------------------------------
// CollisionShape factory methods
//------------------------------------------------------------------------------
CollisionShape CollisionShape::make_sphere(double radius, const datatypes::Transformr& local) {
    // Create a sphere collision shape with given radius
    CollisionShape shape;
    shape.type = CollisionShapeType::SPHERE;
    shape.local_transform = local;
    shape.params.sphere.radius = radius;
    return shape;
}

CollisionShape CollisionShape::make_box(const datatypes::Vector3& half_extents, const datatypes::Transformr& local) {
    // Create a box collision shape with given half extents
    CollisionShape shape;
    shape.type = CollisionShapeType::BOX;
    shape.local_transform = local;
    shape.params.box.half_extents = half_extents;
    return shape;
}

CollisionShape CollisionShape::make_capsule(double radius, double height, const datatypes::Transformr& local) {
    // Create a capsule collision shape (cylinder with hemispherical ends)
    CollisionShape shape;
    shape.type = CollisionShapeType::CAPSULE;
    shape.local_transform = local;
    shape.params.capsule.radius = radius;
    shape.params.capsule.height = height;
    return shape;
}

CollisionShape CollisionShape::make_cylinder(double radius, double height, const datatypes::Transformr& local) {
    // Create a cylinder collision shape
    CollisionShape shape;
    shape.type = CollisionShapeType::CYLINDER;
    shape.local_transform = local;
    shape.params.cylinder.radius = radius;
    shape.params.cylinder.height = height;
    return shape;
}

CollisionShape CollisionShape::make_mesh(std::shared_ptr<Mesh> mesh, const datatypes::Transformr& local) {
    // Create a mesh collision shape (non-convex, expensive)
    CollisionShape shape;
    shape.type = CollisionShapeType::MESH;
    shape.local_transform = local;
    shape.params.mesh.mesh = mesh;
    return shape;
}

CollisionShape CollisionShape::make_convex(const std::vector<datatypes::Vector3>& points, const datatypes::Transformr& local) {
    // Create a convex hull from a set of points
    CollisionShape shape;
    shape.type = CollisionShapeType::CONVEX_HULL;
    shape.local_transform = local;
    shape.params.convex.points = points;
    return shape;
}

CollisionShape CollisionShape::make_plane(const datatypes::Vector3& normal, double offset, const datatypes::Transformr& local) {
    // Create an infinite plane collision shape
    CollisionShape shape;
    shape.type = CollisionShapeType::PLANE;
    shape.local_transform = local;
    shape.params.plane.normal = normal.normalized();
    shape.params.plane.offset = offset;
    return shape;
}

//------------------------------------------------------------------------------
// BaseEntity construction and lifecycle
//------------------------------------------------------------------------------
BaseEntity::BaseEntity()
    : id_(next_id_++)                          // Assign unique incremental ID
    , name_("Entity_" + std::to_string(id_))   // Default name includes ID
    , scene_(nullptr)                          // Not attached to any scene yet
    , transform_()                             // Identity transform
    , initial_transform_()                     // Identity for reset
    , velocity_(0.0)                           // Zero linear velocity
    , angular_vel_(0.0)                        // Zero angular velocity
    , initial_velocity_(0.0)                   // Zero initial velocity
    , initial_angular_vel_(0.0)                // Zero initial angular velocity
    , mass_(1.0)                               // Default mass of 1 kg
    , inv_mass_(1.0)                           // Inverse mass = 1/mass
    , inertia_(1.0, 1.0, 1.0)                  // Default unit sphere inertia
    , inv_inertia_(1.0, 1.0, 1.0)              // Inverse inertia
    , local_com_(0.0)                          // COM at local origin
    , linear_damping_(0.0)                     // No damping by default
    , angular_damping_(0.0)                    // No angular damping
    , force_accum_(0.0)                        // Zero accumulated force
    , torque_accum_(0.0)                       // Zero accumulated torque
    , is_dynamic_(true)                        // Dynamic by default
    , enabled_(true)                           // Enabled by default
    , collidable_(true)                        // Collidable by default
    , raycastable_(true)                       // Raycastable by default
    , aabb_dirty_(true)                        // AABB needs initial compute
{
    // Empty constructor body (initialization list handles everything)
}

BaseEntity::BaseEntity(const std::string& name)
    : BaseEntity()                             // Delegate to default constructor
{
    name_ = name;                              // Override default name with custom
}

BaseEntity::~BaseEntity() {
    // Virtual destructor - required for proper polymorphic destruction
    // No explicit cleanup needed (smart pointers handle resources)
}

BaseEntity::BaseEntity(BaseEntity&& other) noexcept
    : id_(other.id_)                           // Transfer ID
    , name_(std::move(other.name_))            // Move name string
    , scene_(other.scene_)                     // Copy scene pointer
    , transform_(std::move(other.transform_))  // Move transform
    , initial_transform_(std::move(other.initial_transform_))
    , velocity_(std::move(other.velocity_))    // Move velocity
    , angular_vel_(std::move(other.angular_vel_))
    , initial_velocity_(std::move(other.initial_velocity_))
    , initial_angular_vel_(std::move(other.initial_angular_vel_))
    , mass_(other.mass_)                       // Copy scalar mass
    , inv_mass_(other.inv_mass_)
    , inertia_(std::move(other.inertia_))
    , inv_inertia_(std::move(other.inv_inertia_))
    , local_com_(std::move(other.local_com_))
    , linear_damping_(other.linear_damping_)
    , angular_damping_(other.angular_damping_)
    , force_accum_(std::move(other.force_accum_))
    , torque_accum_(std::move(other.torque_accum_))
    , is_dynamic_(other.is_dynamic_)
    , enabled_(other.enabled_)
    , collidable_(other.collidable_)
    , raycastable_(other.raycastable_)
    , shapes_(std::move(other.shapes_))        // Move collision shapes
    , cached_world_aabb_(std::move(other.cached_world_aabb_))
    , aabb_dirty_(other.aabb_dirty_)
{
    // Reset source object to valid but unspecified state
    other.scene_ = nullptr;
    other.shapes_.clear();
    other.aabb_dirty_ = true;
}

BaseEntity& BaseEntity::operator=(BaseEntity&& other) noexcept {
    // Move assignment operator
    if (this != &other) {
        id_ = other.id_;
        name_ = std::move(other.name_);
        scene_ = other.scene_;
        transform_ = std::move(other.transform_);
        initial_transform_ = std::move(other.initial_transform_);
        velocity_ = std::move(other.velocity_);
        angular_vel_ = std::move(other.angular_vel_);
        initial_velocity_ = std::move(other.initial_velocity_);
        initial_angular_vel_ = std::move(other.initial_angular_vel_);
        mass_ = other.mass_;
        inv_mass_ = other.inv_mass_;
        inertia_ = std::move(other.inertia_);
        inv_inertia_ = std::move(other.inv_inertia_);
        local_com_ = std::move(other.local_com_);
        linear_damping_ = other.linear_damping_;
        angular_damping_ = other.angular_damping_;
        force_accum_ = std::move(other.force_accum_);
        torque_accum_ = std::move(other.torque_accum_);
        is_dynamic_ = other.is_dynamic_;
        enabled_ = other.enabled_;
        collidable_ = other.collidable_;
        raycastable_ = other.raycastable_;
        shapes_ = std::move(other.shapes_);
        cached_world_aabb_ = std::move(other.cached_world_aabb_);
        aabb_dirty_ = other.aabb_dirty_;
        
        other.scene_ = nullptr;
        other.shapes_.clear();
        other.aabb_dirty_ = true;
    }
    return *this;
}

//------------------------------------------------------------------------------
// Identification and naming
//------------------------------------------------------------------------------
void BaseEntity::set_name(const std::string& name) {
    name_ = name;                              // Simple assignment (uniqueness enforced by Scene)
}

//------------------------------------------------------------------------------
// Transform and motion state
//------------------------------------------------------------------------------
void BaseEntity::set_transform(const datatypes::Transformr& t) {
    transform_ = t;                            // Set world transform directly
    aabb_dirty_ = true;                        // Invalidate cached AABB
}

datatypes::Vector3 BaseEntity::position() const {
    return transform_.translation;             // Extract translation part
}

void BaseEntity::set_position(const datatypes::Vector3& pos) {
    transform_.translation = pos;              // Set translation only
    aabb_dirty_ = true;                        // AABB changed
}

datatypes::Quat BaseEntity::rotation() const {
    return transform_.rotation;                // Extract rotation part
}

void BaseEntity::set_rotation(const datatypes::Quat& rot) {
    transform_.rotation = rot;                 // Set rotation only
    aabb_dirty_ = true;                        // AABB changed
}

void BaseEntity::translate(const datatypes::Vector3& offset) {
    transform_.translation += offset;          // Add to current translation
    aabb_dirty_ = true;                        // AABB changed
}

void BaseEntity::rotate(const datatypes::Quat& delta) {
    transform_.rotation = delta * transform_.rotation; // Compose rotation on left
    aabb_dirty_ = true;                        // AABB changed
}

void BaseEntity::set_velocity(const datatypes::Vector3& vel) {
    velocity_ = vel;                           // Set linear velocity directly
}

void BaseEntity::set_angular_velocity(const datatypes::Vector3& omega) {
    angular_vel_ = omega;                      // Set angular velocity (axis-angle)
}

//------------------------------------------------------------------------------
// Physics properties
//------------------------------------------------------------------------------
void BaseEntity::set_mass(double m) {
    mass_ = m;                                 // Store mass
    inv_mass_ = (m > 1e-12) ? 1.0 / m : 0.0;   // Compute inverse (0 for static)
    if (inv_mass_ == 0.0) {
        is_dynamic_ = false;                   // Zero mass implies static/kinematic
    }
}

void BaseEntity::set_inertia(const datatypes::Vector3& I) {
    inertia_ = I;                              // Store principal inertias
    inv_inertia_ = datatypes::Vector3(
        (I[0] > 1e-12) ? 1.0 / I[0] : 0.0,     // Inverse of each component
        (I[1] > 1e-12) ? 1.0 / I[1] : 0.0,
        (I[2] > 1e-12) ? 1.0 / I[2] : 0.0
    );
}

datatypes::Vector3 BaseEntity::inverse_inertia() const {
    return inv_inertia_;                       // Return stored inverse
}

datatypes::Matrix3r BaseEntity::world_inertia_tensor() const {
    // Compute inertia tensor in world coordinates: R * I * R^T
    datatypes::Matrix3r R = rotation().toRotationMatrix(); // Rotation matrix
    datatypes::Matrix3r I_local = datatypes::Matrix3r::scale(inertia_); // Diagonal inertia tensor
    return R * I_local * R.transpose();        // Transform to world frame
}

void BaseEntity::set_local_center_of_mass(const datatypes::Vector3& com) {
    local_com_ = com;                          // Set COM offset in local frame
    aabb_dirty_ = true;                        // COM affects world AABB (if shapes are relative)
}

void BaseEntity::set_dynamic(bool dynamic) {
    is_dynamic_ = dynamic;                     // Set dynamic flag
    if (!dynamic) {
        velocity_ = datatypes::Vector3(0.0);   // Static objects have zero velocity
        angular_vel_ = datatypes::Vector3(0.0);
    }
}

void BaseEntity::set_enabled(bool enable) {
    enabled_ = enable;                         // Enable/disable entity in simulation
}

void BaseEntity::set_collidable(bool collidable) {
    collidable_ = collidable;                  // Enable/disable collision participation
}

void BaseEntity::set_raycastable(bool raycastable) {
    raycastable_ = raycastable;                // Enable/disable raycast hits
}

//------------------------------------------------------------------------------
// Collision shapes
//------------------------------------------------------------------------------
void BaseEntity::add_collision_shape(const CollisionShape& shape) {
    shapes_.push_back(shape);                  // Add shape to list
    aabb_dirty_ = true;                        // AABB now includes new shape
}

void BaseEntity::clear_collision_shapes() {
    shapes_.clear();                           // Remove all shapes
    aabb_dirty_ = true;                        // AABB is now empty (or point)
}

datatypes::AABB BaseEntity::world_aabb() const {
    if (aabb_dirty_) {
        update_cached_aabb();                  // Recompute if dirty
    }
    return cached_world_aabb_;                 // Return cached value
}

datatypes::AABB BaseEntity::local_aabb() const {
    datatypes::AABB aabb;                      // Start with empty bounds
    compute_world_aabb(aabb);                  // Compute in world space first
    // Convert world space AABB back to local (approximate, by transforming corners)
    datatypes::Transformr inv = transform_.inverse(); // Inverse transform
    datatypes::AABB local;
    // Expand local AABB by transforming world AABB corners
    datatypes::Vector3 corners[8];
    aabb.compute_corners(corners);             // Get 8 corners of world AABB
    for (const auto& c : corners) {
        local.expand(inv.transformPoint(c));   // Transform to local and expand
    }
    return local;
}

void BaseEntity::update_cached_aabb() const {
    cached_world_aabb_ = datatypes::AABB();    // Reset to empty
    compute_world_aabb(cached_world_aabb_);    // Virtual call to compute bounds
    aabb_dirty_ = false;                       // Mark as clean
}

void BaseEntity::compute_world_aabb(datatypes::AABB& aabb) const {
    // Default implementation: iterate over all shapes and compute their world bounds
    if (shapes_.empty()) {
        // If no shapes, AABB is just a point at entity position (plus margin)
        aabb.expand(position());
        return;
    }
    for (const auto& shape : shapes_) {
        datatypes::Transformr world_tf = transform_ * shape.local_transform; // World transform of shape
        switch (shape.type) {
            case CollisionShapeType::SPHERE: {
                double r = shape.params.sphere.radius;
                datatypes::Vector3 center = world_tf.translation;
                aabb.expand(center - datatypes::Vector3(r, r, r));
                aabb.expand(center + datatypes::Vector3(r, r, r));
                break;
            }
            case CollisionShapeType::BOX: {
                datatypes::Vector3 half = shape.params.box.half_extents;
                // Transform 8 corners of box
                for (int i = 0; i < 8; ++i) {
                    datatypes::Vector3 local_corner(
                        (i & 1) ? half[0] : -half[0],
                        (i & 2) ? half[1] : -half[1],
                        (i & 4) ? half[2] : -half[2]
                    );
                    aabb.expand(world_tf.transformPoint(local_corner));
                }
                break;
            }
            case CollisionShapeType::MESH: {
                if (shape.params.mesh.mesh) {
                    // Transform mesh AABB to world space
                    datatypes::AABB mesh_aabb = shape.params.mesh.mesh->get_aabb();
                    datatypes::AABB world_mesh_aabb;
                    mesh_aabb.transform(world_tf, world_mesh_aabb);
                    aabb.expand(world_mesh_aabb);
                }
                break;
            }
            case CollisionShapeType::PLANE: {
                // Infinite plane: no meaningful finite AABB, just expand around entity
                aabb.expand(world_tf.translation);
                break;
            }
            default:
                // Fallback: expand around shape center
                aabb.expand(world_tf.translation);
                break;
        }
    }
    // Add margin
    if (!shapes_.empty() && shapes_[0].margin > 0) {
        aabb.expand(aabb.min - datatypes::Vector3(shapes_[0].margin));
        aabb.expand(aabb.max + datatypes::Vector3(shapes_[0].margin));
    }
}

//------------------------------------------------------------------------------
// Forces and impulses
//------------------------------------------------------------------------------
void BaseEntity::apply_force(const datatypes::Vector3& force) {
    // Apply force at center of mass (no torque)
    force_accum_ += force;                     // Accumulate in world space
}

void BaseEntity::apply_force(const datatypes::Vector3& force, const datatypes::Vector3& world_point) {
    // Apply force at arbitrary point (causes torque as well)
    force_accum_ += force;                     // Add to linear force accumulator
    datatypes::Vector3 r = world_point - transform_.transformPoint(local_com_); // Vector from COM to point
    torque_accum_ += r.cross(force);           // Torque = r × F
}

void BaseEntity::apply_torque(const datatypes::Vector3& torque) {
    torque_accum_ += torque;                   // Accumulate torque in world space
}

void BaseEntity::apply_impulse(const datatypes::Vector3& impulse) {
    // Instantaneously change linear velocity (impulse = change in momentum)
    if (inv_mass_ > 0.0) {
        velocity_ += impulse * inv_mass_;      // Δv = J / m
    }
}

void BaseEntity::apply_impulse(const datatypes::Vector3& impulse, const datatypes::Vector3& world_point) {
    // Apply impulse at point: changes both linear and angular velocity
    if (inv_mass_ > 0.0) {
        velocity_ += impulse * inv_mass_;      // Linear change
        datatypes::Vector3 r = world_point - transform_.transformPoint(local_com_); // Vector from COM
        // Angular impulse: ω += I^{-1} (r × J)
        datatypes::Vector3 angular_impulse = r.cross(impulse);
        datatypes::Matrix3r I_world = world_inertia_tensor();
        // Solve I * Δω = angular_impulse (approximate using diagonal for simplicity)
        // In full implementation we would use the full inertia tensor inverse
        angular_vel_ += datatypes::Vector3(
            angular_impulse[0] * inv_inertia_[0],
            angular_impulse[1] * inv_inertia_[1],
            angular_impulse[2] * inv_inertia_[2]
        );
    }
}

void BaseEntity::clear_forces() {
    force_accum_ = datatypes::Vector3(0.0);    // Reset force accumulator
    torque_accum_ = datatypes::Vector3(0.0);   // Reset torque accumulator
}

//------------------------------------------------------------------------------
// Integration (explicit Euler by default, can be overridden)
//------------------------------------------------------------------------------
void BaseEntity::integrate(double dt) {
    if (!is_dynamic_ || !enabled_) return;     // Only integrate dynamic, enabled entities
    integrate_velocity(dt);                    // Update velocity from forces
    integrate_position(dt);                    // Update position from velocity
}

void BaseEntity::integrate_velocity(double dt) {
    // Compute acceleration from accumulated forces
    datatypes::Vector3 acceleration = force_accum_ * inv_mass_; // a = F / m
    velocity_ += acceleration * dt;            // v += a * dt
    // Apply linear damping
    velocity_ *= (1.0 - linear_damping_ * dt);
    
    // Compute angular acceleration (simplified: using diagonal inertia)
    datatypes::Vector3 angular_accel(
        torque_accum_[0] * inv_inertia_[0],
        torque_accum_[1] * inv_inertia_[1],
        torque_accum_[2] * inv_inertia_[2]
    );
    angular_vel_ += angular_accel * dt;        // ω += α * dt
    // Apply angular damping
    angular_vel_ *= (1.0 - angular_damping_ * dt);
}

void BaseEntity::integrate_position(double dt) {
    // Update translation: x += v * dt
    transform_.translation += velocity_ * dt;
    // Update rotation: q += 0.5 * dt * (0, ω) * q  (quaternion derivative)
    if (angular_vel_.squaredNorm() > 1e-12) {
        double angle = angular_vel_.norm() * dt; // Rotation angle
        datatypes::Vector3 axis = angular_vel_.normalized(); // Rotation axis
        datatypes::Quat delta_q(axis, angle);   // Quaternion representing rotation
        transform_.rotation = delta_q * transform_.rotation; // Apply rotation
        transform_.rotation.normalize();         // Ensure unit quaternion
    }
    aabb_dirty_ = true;                        // Position changed, invalidate AABB
}

//------------------------------------------------------------------------------
// Queries
//------------------------------------------------------------------------------
datatypes::Vector3 BaseEntity::velocity_at_point(const datatypes::Vector3& world_point) const {
    // Linear velocity plus tangential velocity due to rotation: v + ω × r
    datatypes::Vector3 com = transform_.transformPoint(local_com_); // World COM
    datatypes::Vector3 r = world_point - com;   // Vector from COM to point
    return velocity_ + angular_vel_.cross(r);   // v_point = v + ω × r
}

double BaseEntity::inverse_mass(const datatypes::Vector3& world_point, const datatypes::Vector3& normal) const {
    // Effective inverse mass along a given normal at a point
    if (inv_mass_ == 0.0) return 0.0;          // Static object has infinite mass
    double w = inv_mass_;                      // Start with linear part
    if (angular_vel_.squaredNorm() == 0.0 && inv_inertia_.max() == 0.0) return w;
    datatypes::Vector3 r = world_point - transform_.transformPoint(local_com_); // Lever arm
    datatypes::Vector3 n_cross_r = normal.cross(r); // n × r
    // Angular contribution: (r × n)^T * I^{-1} * (r × n)
    // Simplification using diagonal inertia in world frame (approximate)
    datatypes::Vector3 r_cross_n = r.cross(normal);
    double ang_inv = r_cross_n[0] * r_cross_n[0] * inv_inertia_[0] +
                     r_cross_n[1] * r_cross_n[1] * inv_inertia_[1] +
                     r_cross_n[2] * r_cross_n[2] * inv_inertia_[2];
    return w + ang_inv;                        // Total effective inverse mass
}

bool BaseEntity::collide_with(const BaseEntity& other, double dt, std::vector<ContactPoint>& contacts) const {
    // Default narrow-phase collision (should be overridden by concrete types)
    // For now, just do AABB overlap test as a placeholder
    if (!collidable_ || !other.collidable_) return false;
    datatypes::AABB a = world_aabb();
    datatypes::AABB b = other.world_aabb();
    if (!a.intersects(b)) return false;        // No AABB overlap
    // Subclasses should implement proper shape-based collision detection
    return false;                              // Return false by default (no implementation)
}

RayHit BaseEntity::ray_cast(const datatypes::Ray& ray) const {
    RayHit hit;                                // Default: no hit
    if (!raycastable_) return hit;
    
    // Transform ray to local space for shape testing
    datatypes::Transformr inv_transform = transform_.inverse();
    datatypes::Ray local_ray(inv_transform.transformPoint(ray.origin),
                             inv_transform.transformVector(ray.direction));
    
    double closest_t = std::numeric_limits<double>::max();
    for (const auto& shape : shapes_) {
        double t = std::numeric_limits<double>::max();
        datatypes::Vector3 normal;
        bool shape_hit = false;
        // Basic ray-shape intersection (simplified)
        switch (shape.type) {
            case CollisionShapeType::SPHERE: {
                double r = shape.params.sphere.radius;
                datatypes::Vector3 oc = local_ray.origin - shape.local_transform.translation;
                double b = oc.dot(local_ray.direction);
                double c = oc.squaredNorm() - r * r;
                double disc = b*b - c;
                if (disc >= 0) {
                    double sqrt_disc = std::sqrt(disc);
                    double t0 = -b - sqrt_disc;
                    double t1 = -b + sqrt_disc;
                    if (t0 > 1e-6) {
                        t = t0;
                        shape_hit = true;
                    } else if (t1 > 1e-6) {
                        t = t1;
                        shape_hit = true;
                    }
                    if (shape_hit) {
                        datatypes::Vector3 hit_local = local_ray.pointAt(t);
                        normal = (hit_local - shape.local_transform.translation).normalized();
                    }
                }
                break;
            }
            case CollisionShapeType::PLANE: {
                datatypes::Vector3 n = shape.params.plane.normal;
                double d = shape.params.plane.offset;
                double denom = n.dot(local_ray.direction);
                if (std::abs(denom) > 1e-12) {
                    t = -(n.dot(local_ray.origin) + d) / denom;
                    if (t > 1e-6) {
                        shape_hit = true;
                        normal = n;
                    }
                }
                break;
            }
            default:
                // Other shapes not implemented in base
                break;
        }
        if (shape_hit && t < closest_t) {
            closest_t = t;
            hit.hit = true;
            hit.t = t;
            hit.point = ray.pointAt(t);        // Convert back to world space
            hit.normal = transform_.transformVector(normal).normalized();
            hit.entity_id = id_;
        }
    }
    return hit;
}

//------------------------------------------------------------------------------
// Scene linkage
//------------------------------------------------------------------------------
void BaseEntity::set_scene(Scene* scene) {
    scene_ = scene;                            // Store weak pointer to owning scene
}

//------------------------------------------------------------------------------
// Reset
//------------------------------------------------------------------------------
void BaseEntity::reset() {
    transform_ = initial_transform_;           // Restore initial transform
    velocity_ = initial_velocity_;             // Restore initial velocity
    angular_vel_ = initial_angular_vel_;       // Restore initial angular velocity
    clear_forces();                            // Clear accumulated forces
    aabb_dirty_ = true;                        // Invalidate AABB cache
}

//------------------------------------------------------------------------------
// String representation
//------------------------------------------------------------------------------
std::string BaseEntity::repr() const {
    std::ostringstream oss;
    oss << "BaseEntity(id=" << id_
        << ", name=\"" << name_ << "\""
        << ", pos=" << position()
        << ", mass=" << mass_
        << ", dynamic=" << is_dynamic_
        << ")";
    return oss.str();
}

std::string BaseEntity::str() const {
    return name_ + " (id=" + std::to_string(id_) + ")"; // Human-readable short string
}

} // namespace engine
} // namespace genesis