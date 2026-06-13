// File 206: modules/newton/src/servers/newton_physics_server_3d.cpp
// Implementación del servidor de física NewtonPhysicsServer3D que reemplaza
// el motor de física por defecto de Godot con Newton Dynamics 4.0.

#include "newton_physics_server_3d.h"

// Newton core
#include "../world/newton_world.h"
#include "../bodies/newton_body.h"
#include "../joints/newton_joint.h"
#include "../joints/newton_hinge_joint.h"
#include "../joints/newton_slider_joint.h"
#include "../joints/newton_universal_joint.h"
#include "../joints/newton_corkscrew_joint.h"
#include "../joints/newton_kinematic_controller.h"
#include "../collision/newton_collision.h"
#include "../collision/newton_compound_collision.h"
#include "../materials/newton_material.h"
#include "../vehicles/newton_vehicle.h"

// Gaia broad‑phase
#include "../../../gaia/src/collision_detector/broad_phase.h"

// Godot
#include "core/config/project_settings.h"
#include "core/io/json.h"
#include "core/object/class_db.h"
#include "core/string/ustring.h"
#include "core/variant/variant.h"
#include "core/typedefs.h"

// Constructor: crea el mundo Newton y ajusta valores por defecto.
NewtonPhysicsServer3D::NewtonPhysicsServer3D() :
	active(false),
	step_size(1.0 / 60.0),
	next_body_id(1),
	next_joint_id(1),
	next_material_id(1),
	gravity(0.0, -9.80665, 0.0) {
	// Instancia el mundo Newton que orquesta la simulación.
	newton_world = memnew(newton::NewtonWorld);
	newton_world->set_gravity(gravity);
	newton_world->set_solver_iterations(16);
}

NewtonPhysicsServer3D::~NewtonPhysicsServer3D() {
	if (newton_world) {
		memdelete(newton_world);
		newton_world = nullptr;
	}
}

bool NewtonPhysicsServer3D::is_flushing_queries() const {
	return false;
}

int NewtonPhysicsServer3D::get_process_info(ProcessInfo p_info) {
	// Devuelve información del proceso de física.
	switch (p_info) {
		case INFO_ACTIVE_OBJECTS: return newton_world ? newton_world->get_body_count() : 0;
		case INFO_COLLISION_PAIRS: return 0; // no expuesto aún
		case INFO_ISLAND_COUNT: return 0;
		default: return 0;
	}
}

RID NewtonPhysicsServer3D::space_create() {
	// Newton maneja un único espacio mundial; se devuelve RID vacío.
	return RID();
}

RID NewtonPhysicsServer3D::area_create() {
	return RID();
}

// Crea un cuerpo rígido en Newton.
RID NewtonPhysicsServer3D::body_create() {
	RID rid = RID(); // generará un RID único internamente mediante el owner.
	BodyData bd;
	bd.self = rid;
	bd.body_type = newton::BodyType::DYNAMIC;
	bd.active = true;
	// Crea el cuerpo Newton subyacente.
	Ref<newton::NewtonBody> nb;
	nb.instantiate();
	bd.newton_body = nb;
	// Inserta el cuerpo en el mundo Newton.
	newton::body_id nid = newton_world->create_body(nb);
	bd.nid = nid;
	body_map[rid] = bd;
	newton_to_rid[nid] = rid;
	return rid;
}

RID NewtonPhysicsServer3D::soft_body_create() {
	// Newton no tiene soft‑bodies nativos; se puede delegar en Gaia/Genesis.
	// Por ahora retornamos un RID normal de cuerpo rígido.
	return body_create();
}

RID NewtonPhysicsServer3D::shape_create(ShapeType p_type) {
	// Crea una forma de colisión Newton según el tipo Godot.
	Ref<newton::NewtonCollision> coll;
	switch (p_type) {
		case SHAPE_SPHERE: coll.instantiate(); break; // NewtonCollisionSphere se crea con radio por defecto.
		case SHAPE_BOX: coll.instantiate(); break;    // NewtonCollisionBox
		case SHAPE_CAPSULE: coll.instantiate(); break;// NewtonCollisionCapsule
		case SHAPE_CYLINDER: coll.instantiate(); break;// NewtonCollisionCylinder
		default: coll.instantiate(); break;           // caja por defecto
	}
	RID rid = RID();
	shape_map[rid] = coll;
	return rid;
}

void NewtonPhysicsServer3D::body_set_space(RID p_body, RID p_space) {
	// Newton no maneja espacios separados; se ignora.
}

void NewtonPhysicsServer3D::body_set_mode(RID p_body, BodyMode p_mode) {
	HashMap<RID, BodyData>::Iterator it = body_map.find(p_body);
	if (!it) return;
	switch (p_mode) {
		case BODY_MODE_STATIC: it->value.body_type = newton::BodyType::STATIC; break;
		case BODY_MODE_KINEMATIC: it->value.body_type = newton::BodyType::KINEMATIC; break;
		case BODY_MODE_RIGID:
		default: it->value.body_type = newton::BodyType::DYNAMIC; break;
	}
	if (it->value.newton_body.is_valid()) {
		it->value.newton_body->set_type(it->value.body_type);
	}
}

void NewtonPhysicsServer3D::body_set_state(RID p_body, BodyState p_state, const Variant &p_value) {
	HashMap<RID, BodyData>::Iterator it = body_map.find(p_body);
	if (!it) return;
	Ref<newton::NewtonBody> &nb = it->value.newton_body;
	if (nb.is_null()) return;

	switch (p_state) {
		case BODY_STATE_TRANSFORM: {
			nb->set_transform(p_value);
		} break;
		case BODY_STATE_LINEAR_VELOCITY: {
			nb->set_linear_velocity(p_value);
		} break;
		case BODY_STATE_ANGULAR_VELOCITY: {
			nb->set_angular_velocity(p_value);
		} break;
		case BODY_STATE_SLEEPING: {
			bool sleep = p_value;
			nb->set_active(!sleep);
		} break;
		case BODY_STATE_CAN_SLEEP: {
			nb->set_active(true); // Newton maneja sleep automáticamente.
		} break;
		default: break;
	}
}

Variant NewtonPhysicsServer3D::body_get_state(RID p_body, BodyState p_state) {
	HashMap<RID, BodyData>::Iterator it = body_map.find(p_body);
	ERR_FAIL_COND_V(!it, Variant());
	Ref<newton::NewtonBody> &nb = it->value.newton_body;
	if (nb.is_null()) return Variant();

	switch (p_state) {
		case BODY_STATE_TRANSFORM: return nb->get_transform();
		case BODY_STATE_LINEAR_VELOCITY: return nb->get_linear_velocity();
		case BODY_STATE_ANGULAR_VELOCITY: return nb->get_angular_velocity();
		case BODY_STATE_SLEEPING: return !nb->is_active();
		default: return Variant();
	}
}

void NewtonPhysicsServer3D::body_add_shape(RID p_body, RID p_shape, const Transform3D &p_transform, bool p_disabled) {
	HashMap<RID, BodyData>::Iterator it = body_map.find(p_body);
	if (!it) return;
	HashMap<RID, Ref<newton::NewtonCollision>>::Iterator s_it = shape_map.find(p_shape);
	if (s_it == shape_map.end()) return;

	// Asocia la forma al cuerpo. Si ya existe una forma compuesta, la agrega.
	// Para simplicidad asumimos una única forma por cuerpo.
	it->value.newton_collision = s_it->value;
	// Calcula inercias a partir de la masa actual.
	if (it->value.newton_body.is_valid() && s_it->value.is_valid()) {
		real_t mass = it->value.newton_body->get_mass();
		mat3 inertia = s_it->value->compute_inertia(mass);
		it->value.newton_body->set_inertia(inertia);
		// Actualiza el AABB a partir de la forma.
		aabb local_aabb = s_it->value->get_local_aabb();
		it->value.newton_body->set_collision_aabb(local_aabb);
	}
}

void NewtonPhysicsServer3D::body_set_shape(RID p_body, int p_shape_idx, RID p_shape) {
	// Newton no maneja múltiples shapes por índice de manera directa.
	// Por simplicidad ignoramos el índice y asignamos la forma.
	body_add_shape(p_body, p_shape, Transform3D(), false);
}

void NewtonPhysicsServer3D::body_set_shape_transform(RID p_body, int p_shape_idx, const Transform3D &p_transform) {
	// Newton no permite modificar el offset de la forma tras la creación.
	// Se debe usar una forma compuesta. Por ahora ignoramos.
}

void NewtonPhysicsServer3D::physics_step(real_t p_step) {
	if (!active || !newton_world) return;

	// Sincroniza las transformaciones de los cuerpos Godot → Newton.
	for (KeyValue<RID, BodyData> &kv : body_map) {
		_sync_body_to_newton(kv.key);
	}

	// Ejecuta un paso del mundo Newton.
	newton_world->step(p_step);

	// Sincroniza Newton → Godot.
	for (KeyValue<newton::body_id, RID> &kv : newton_to_rid) {
		_sync_newton_to_body(kv.key);
	}
}

void NewtonPhysicsServer3D::set_active(bool p_active) {
	active = p_active;
}

bool NewtonPhysicsServer3D::is_active() const {
	return active;
}

void NewtonPhysicsServer3D::load_options(const String &path) {
	// Carga parámetros de un archivo JSON y los aplica al mundo Newton.
	Ref<FileAccess> f = FileAccess::open(path, FileAccess::READ);
	if (f.is_null()) return;
	String json_text = f->get_as_utf8_string();
	JSON json;
	Error err = json.parse(json_text);
	if (err != OK) return;
	Dictionary opts = json.get_data();
	// Aplica opciones conocidas.
	if (opts.has("gravity")) {
		Array g = opts["gravity"];
		if (g.size() == 3) {
			gravity = Vector3(g[0], g[1], g[2]);
			newton_world->set_gravity(gravity);
		}
	}
	if (opts.has("solver_iterations")) {
		newton_world->set_solver_iterations(opts["solver_iterations"]);
	}
	if (opts.has("sleep_frames")) {
		newton_world->set_sleep_frames(opts["sleep_frames"]);
	}
}

void NewtonPhysicsServer3D::_sync_body_to_newton(const RID &p_body) {
	HashMap<RID, BodyData>::Iterator it = body_map.find(p_body);
	if (!it) return;
	Ref<newton::NewtonBody> &nb = it->value.newton_body;
	if (nb.is_null() || !it->value.active) return;

	// La transformación ya se actualiza a través de body_set_state.
	// Si el cuerpo es kinematic, su velocidad se calcula automáticamente.
}

void NewtonPhysicsServer3D::_sync_newton_to_body(newton::body_id p_nid) {
	HashMap<newton::body_id, RID>::Iterator it = newton_to_rid.find(p_nid);
	if (!it) return;
	RID rid = it->value;
	HashMap<RID, BodyData>::Iterator b_it = body_map.find(rid);
	if (!b_it) return;
	Ref<newton::NewtonBody> &nb = b_it->value.newton_body;
	if (nb.is_null()) return;

	// La lógica de Godot espera leer el estado del cuerpo; no almacenamos nada
	// adicional ya que body_get_state consulta directamente el NewtonBody.
}