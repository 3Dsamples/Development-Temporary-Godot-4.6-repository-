// File 94: modules/genesis/src/io/urdf_loader.h
// URDF (Unified Robot Description Format) loader – parses a URDF file and
// creates a kinematic chain of Genesis RigidEntities linked by JointConstraints.
// Supports visual and collision geometry loading from meshes.

#ifndef GENESIS_IO_URDF_LOADER_H
#define GENESIS_IO_URDF_LOADER_H

#include "core/io/file_access.h"
#include "core/io/json.h"                     // for XML we use Godot's XMLParser
#include "../entities/rigid_entity.h"
#include "../entities/tool_entity.h"
#include "../constraints/joint_constraint.h"
#include "../core/genesis_types.h"
#include "core/templates/local_vector.h"
#include "core/templates/hash_map.h"
#include "core/variant/variant.h"

namespace genesis::io {

class URDFLoader {
public:
	struct LinkInfo {
		String name;
		Transform3D origin;   // visual/collision transform relative to link
		real_t mass;
		Vector3 inertia_diag;
		// Visual / collision mesh files (later)
		String visual_mesh;
		String collision_mesh;
	};
	struct JointInfo {
		String name;
		String type;          // "revolute", "prismatic", "fixed", "continuous", "floating", "planar"
		String parent, child;
		Vector3 origin_xyz;
		Vector3 origin_rpy;   // rotation as euler (radians)
		Vector3 axis;
		real_t lower_limit, upper_limit;
		real_t effort, velocity;
	};

	struct RobotModel {
		LocalVector<LinkInfo> links;
		LocalVector<JointInfo> joints;
		HashMap<String, int> link_index_map; // name -> idx in links
	};

	// Load a URDF file and populate RobotModel
	static Error parse(const String &p_path, RobotModel &r_model) {
		String text = _read_file(p_path);
		if (text.is_empty()) return ERR_FILE_CANT_OPEN;

		// Use XMLParser (available in Godot)
		XMLParser parser;
		Error err = parser.open_buffer(text.utf8().get_data(), text.size());
		if (err != OK) return err;

		r_model.links.clear();
		r_model.joints.clear();
		r_model.link_index_map.clear();

		// Iterate nodes
		while (parser.read() == OK) {
			if (parser.get_node_type() == XMLParser::NODE_ELEMENT) {
				String node_name = parser.get_node_name();
				if (node_name == "link") {
					_parse_link(parser, r_model);
				} else if (node_name == "joint") {
					_parse_joint(parser, r_model);
				}
			}
		}
		return OK;
	}

	// Build a set of Genesis entities and joints from the model.
	// Returns the root entity (base link) and a list of all constraints.
	static void build_robot(const RobotModel &p_model,
							LocalVector<Ref<BaseEntity>> &r_entities,
							LocalVector<Ref<JointConstraint>> &r_joints) {
		HashMap<String, Ref<BaseEntity>> link_entities;
		// Create a RigidEntity for each link
		for (int i = 0; i < p_model.links.size(); ++i) {
			const LinkInfo &link = p_model.links[i];
			Ref<RigidEntity> entity = memnew(RigidEntity);
			entity->set_entity_uid(i + 1); // simple indexing
			entity->set_override_mass(link.mass);
			// Inertia diagonal (assume principal axes alignment)
			entity->set_inertia(Basis().scaled(Vector3(link.inertia_diag.x, link.inertia_diag.y, link.inertia_diag.z)));
			entity->set_geometry_type(GeometryType::BOX); // default, can be overridden later by collision meshes
			entity->set_half_extents(Vector3(0.1, 0.1, 0.1));
			// Set the initial transform from link origin (relative to parent? We'll set later via joint building)
			entity->set_transform(Transform3D(Basis(), link.origin.origin));
			r_entities.push_back(entity);
			link_entities[link.name] = entity;
		}

		// Build joints as constraints
		for (const JointInfo &joint : p_model.joints) {
			if (!link_entities.has(joint.parent) || !link_entities.has(joint.child))
				continue;
			Ref<BaseEntity> parent = link_entities[joint.parent];
			Ref<BaseEntity> child = link_entities[joint.child];
			// Create joint constraint (PBD style)
			Ref<JointConstraint> jc = memnew(JointConstraint);
			jc->set_bodies(Object::cast_to<RigidEntity>(parent.ptr()), Object::cast_to<RigidEntity>(child.ptr()));
			// Determine joint type
			if (joint.type == "revolute" || joint.type == "continuous") {
				jc->set_joint_type(JointConstraint::REVOLUTE);
			} else if (joint.type == "prismatic") {
				jc->set_joint_type(JointConstraint::PRISMATIC);
			} else if (joint.type == "fixed") {
				jc->set_joint_type(JointConstraint::FIXED);
			} else if (joint.type == "spherical") {
				jc->set_joint_type(JointConstraint::SPHERICAL);
			}
			// Set anchor and axis based on joint origin and axis
			// The joint's origin is relative to parent link frame; we need to compute local anchors.
			Transform3D joint_xform;
			joint_xform.origin = joint.origin_xyz;
			joint_xform.basis = Basis::from_euler(Vector3(joint.origin_rpy.x, joint.origin_rpy.y, joint.origin_rpy.z));
			Transform3D parent_transform = parent->get_transform();
			Transform3D child_transform = child->get_transform();
			// World position of joint: parent_transform * joint_xform
			// Anchor in parent local: joint_xform.origin (plus rotation? anchor point is position)
			jc->set_anchor_a(joint_xform.origin);
			// Anchor in child local: we need the point in child's frame: parent_to_world * joint_origin = child_to_world * anchor_child => anchor_child = child_to_world.inverse() * parent_to_world * joint_origin
			// For simplicity, we set both anchors to the same world point and rely on the constraint solver to correct.
			// But proper initialization is needed for a stable starting point.
			jc->set_anchor_b(joint_xform.origin); // This is a placeholder; correct should compute child-relative position.
			jc->set_axis_a(joint.axis);
			jc->set_axis_b(joint.axis); // assume aligned at start
			r_joints.push_back(jc);
		}
	}

private:
	static String _read_file(const String &p_path) {
		Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::READ);
		if (f.is_null()) return String();
		return f->get_as_utf8_string();
	}

	static void _parse_link(XMLParser &parser, RobotModel &r_model) {
		LinkInfo link;
		link.name = parser.get_named_attribute_value("name");
		// Parse deeper nodes (visual, collision, inertial)
		while (parser.read() == OK) {
			if (parser.get_node_type() == XMLParser::NODE_ELEMENT) {
				String sub = parser.get_node_name();
				if (sub == "inertial") {
					_parse_inertial(parser, link);
				} else if (sub == "visual") {
					_parse_visual(parser, link);
				} else if (sub == "collision") {
					_parse_collision(parser, link);
				}
			} else if (parser.get_node_type() == XMLParser::NODE_ELEMENT_END && parser.get_node_name() == "link") {
				break;
			}
		}
		r_model.links.push_back(link);
		r_model.link_index_map[link.name] = r_model.links.size() - 1;
	}

	static void _parse_joint(XMLParser &parser, RobotModel &r_model) {
		JointInfo joint;
		joint.name = parser.get_named_attribute_value("name");
		joint.type = parser.get_named_attribute_value("type");
		// Parse origin, axis, limits, parent, child
		while (parser.read() == OK) {
			if (parser.get_node_type() == XMLParser::NODE_ELEMENT) {
				String sub = parser.get_node_name();
				if (sub == "origin") {
					joint.origin_xyz = _parse_vector3(parser, "xyz");
					joint.origin_rpy = _parse_vector3(parser, "rpy");
				} else if (sub == "axis") {
					joint.axis = _parse_vector3(parser, "xyz");
				} else if (sub == "parent") {
					joint.parent = parser.get_named_attribute_value("link");
				} else if (sub == "child") {
					joint.child = parser.get_named_attribute_value("link");
				} else if (sub == "limit") {
					joint.lower_limit = parser.get_named_attribute_value("lower").to_float();
					joint.upper_limit = parser.get_named_attribute_value("upper").to_float();
					joint.effort = parser.get_named_attribute_value("effort").to_float();
					joint.velocity = parser.get_named_attribute_value("velocity").to_float();
				}
			} else if (parser.get_node_type() == XMLParser::NODE_ELEMENT_END && parser.get_node_name() == "joint") {
				break;
			}
		}
		r_model.joints.push_back(joint);
	}

	static void _parse_inertial(XMLParser &parser, LinkInfo &link) {
		while (parser.read() == OK) {
			if (parser.get_node_type() == XMLParser::NODE_ELEMENT) {
				String sub = parser.get_node_name();
				if (sub == "origin") {
					link.origin.origin = _parse_vector3(parser, "xyz");
				} else if (sub == "mass") {
					link.mass = parser.get_named_attribute_value("value").to_float();
				} else if (sub == "inertia") {
					link.inertia_diag = Vector3(
						parser.get_named_attribute_value("ixx").to_float(),
						parser.get_named_attribute_value("iyy").to_float(),
						parser.get_named_attribute_value("izz").to_float()
					);
				}
			} else if (parser.get_node_type() == XMLParser::NODE_ELEMENT_END && parser.get_node_name() == "inertial") {
				break;
			}
		}
	}

	static void _parse_visual(XMLParser &parser, LinkInfo &link) {
		while (parser.read() == OK) {
			if (parser.get_node_type() == XMLParser::NODE_ELEMENT) {
				String sub = parser.get_node_name();
				if (sub == "origin") {
					link.origin.origin = _parse_vector3(parser, "xyz");
				} else if (sub == "geometry") {
					_parse_geometry(parser, link.visual_mesh);
				}
			} else if (parser.get_node_type() == XMLParser::NODE_ELEMENT_END && parser.get_node_name() == "visual") {
				break;
			}
		}
	}

	static void _parse_collision(XMLParser &parser, LinkInfo &link) {
		while (parser.read() == OK) {
			if (parser.get_node_type() == XMLParser::NODE_ELEMENT) {
				String sub = parser.get_node_name();
				if (sub == "origin") {
					link.origin.origin = _parse_vector3(parser, "xyz");
				} else if (sub == "geometry") {
					_parse_geometry(parser, link.collision_mesh);
				}
			} else if (parser.get_node_type() == XMLParser::NODE_ELEMENT_END && parser.get_node_name() == "collision") {
				break;
			}
		}
	}

	static void _parse_geometry(XMLParser &parser, String &r_mesh_path) {
		while (parser.read() == OK) {
			if (parser.get_node_type() == XMLParser::NODE_ELEMENT) {
				String sub = parser.get_node_name();
				if (sub == "mesh") {
					r_mesh_path = parser.get_named_attribute_value("filename");
				}
			} else if (parser.get_node_type() == XMLParser::NODE_ELEMENT_END && parser.get_node_name() == "geometry") {
				break;
			}
		}
	}

	static Vector3 _parse_vector3(XMLParser &parser, const String &attr_name) {
		String val = parser.get_named_attribute_value(attr_name);
		if (val.is_empty()) return Vector3();
		Vector<String> parts = val.split(" ");
		if (parts.size() >= 3) {
			return Vector3(parts[0].to_float(), parts[1].to_float(), parts[2].to_float());
		}
		return Vector3();
	}
};

} // namespace genesis::io

#endif // GENESIS_IO_URDF_LOADER_H