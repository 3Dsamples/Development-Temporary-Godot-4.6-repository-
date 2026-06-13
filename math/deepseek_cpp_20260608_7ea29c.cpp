// File 35: modules/gaia/src/mesh/mesh_io.h

#ifndef GAIA_MESH_IO_H
#define GAIA_MESH_IO_H

#include "tet_mesh.h"
#include "tri_mesh.h"

#include "core/io/file_access.h"
#include "core/string/ustring.h"
#include "core/templates/local_vector.h"
#include "core/variant/variant.h"

namespace gaia::mesh {

/**
 * Mesh I/O – loading triangle and tetrahedral meshes from common formats.
 * Currently supports Wavefront OBJ (triangles) and TetGen .node/.ele pairs.
 */
class MeshIO {
public:
    /**
     * Load an OBJ file into a TriMesh. Only vertices (v) and faces (f) are read.
     * Lines starting with # are skipped.
     * Returns OK on success, or an error code.
     */
    static Error load_obj(const String &p_path, TriMesh &r_mesh) {
        Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::READ);
        ERR_FAIL_COND_V_MSG(file.is_null(), ERR_FILE_CANT_OPEN, "Cannot open OBJ: " + p_path);

        r_mesh.clear();
        LocalVector<Vector3> temp_vertices;
        LocalVector<TriMesh::Triangle> temp_triangles;

        while (!file->eof_reached()) {
            String line = file->get_line().strip_edges();
            if (line.is_empty() || line.begins_with("#")) continue;

            Vector<String> parts = line.split(" ", false);
            if (parts.is_empty()) continue;

            String cmd = parts[0];
            if (cmd == "v" && parts.size() >= 4) {
                Vector3 v;
                v.x = parts[1].to_float();
                v.y = parts[2].to_float();
                v.z = parts[3].to_float();
                temp_vertices.push_back(v);
            } else if (cmd == "f" && parts.size() >= 4) {
                TriMesh::Triangle tri;
                // Face indices can be like "1" or "1/2/3" or "1//2".
                tri.v0 = parse_face_index(parts[1]) - 1; // OBJ is 1-based
                tri.v1 = parse_face_index(parts[2]) - 1;
                tri.v2 = parse_face_index(parts[3]) - 1;
                if (tri.v0 < 0 || tri.v1 < 0 || tri.v2 < 0) {
                    ERR_CONTINUE_MSG(true, "Invalid face indices in line: " + line);
                }
                temp_triangles.push_back(tri);
            }
        }

        // Build TriMesh
        for (const Vector3 &v : temp_vertices) {
            r_mesh.add_vertex(v);
        }
        for (const TriMesh::Triangle &t : temp_triangles) {
            r_mesh.add_triangle(t.v0, t.v1, t.v2);
        }

        if (r_mesh.vertex_count() == 0 && r_mesh.triangle_count() == 0) {
            return ERR_PARSE_ERROR;
        }
        return OK;
    }

    /**
     * Load a TetGen .node and .ele pair into a TetMesh.
     * .node: first line: <num_vertices> <dim> <num_attrs> <boundary_marker>
     *         following lines: index x y z [attrs ...] [boundary]
     * .ele:  first line: <num_tets> <nodes_per_tet> <num_attrs>
     *         following lines: index v1 v2 v3 v4 [attr]
     */
    static Error load_tetgen(const String &p_node_path, const String &p_ele_path, TetMesh &r_mesh) {
        // Load .node
        Ref<FileAccess> node_file = FileAccess::open(p_node_path, FileAccess::READ);
        ERR_FAIL_COND_V_MSG(node_file.is_null(), ERR_FILE_CANT_OPEN, "Cannot open .node: " + p_node_path);
        // Read header
        String header = node_file->get_line().strip_edges();
        Vector<String> hparts = header.split(" ", false);
        if (hparts.size() < 2) return ERR_PARSE_ERROR;
        int num_vertices = hparts[0].to_int();
        int dim = hparts[1].to_int();
        if (num_vertices <= 0 || dim != 3) return ERR_PARSE_ERROR;

        r_mesh.clear();
        for (int i = 0; i < num_vertices; ++i) {
            String line = node_file->get_line().strip_edges();
            Vector<String> parts = line.split(" ", false);
            if (parts.size() < 4) continue; // skip incomplete
            // first part is index, skip
            Vector3 v;
            v.x = parts[1].to_float();
            v.y = parts[2].to_float();
            v.z = parts[3].to_float();
            r_mesh.add_vertex(v);
        }
        node_file.unref();

        // Load .ele
        Ref<FileAccess> ele_file = FileAccess::open(p_ele_path, FileAccess::READ);
        ERR_FAIL_COND_V_MSG(ele_file.is_null(), ERR_FILE_CANT_OPEN, "Cannot open .ele: " + p_ele_path);
        header = ele_file->get_line().strip_edges();
        hparts = header.split(" ", false);
        if (hparts.size() < 2) return ERR_PARSE_ERROR;
        int num_tets = hparts[0].to_int();
        int nodes_per_tet = hparts[1].to_int();
        if (num_tets <= 0 || nodes_per_tet != 4) return ERR_PARSE_ERROR;

        for (int i = 0; i < num_tets; ++i) {
            String line = ele_file->get_line().strip_edges();
            Vector<String> parts = line.split(" ", false);
            if (parts.size() < 5) continue;
            int idx = parts[0].to_int();
            int v0 = parts[1].to_int() - 1; // TetGen indices are 1-based
            int v1 = parts[2].to_int() - 1;
            int v2 = parts[3].to_int() - 1;
            int v3 = parts[4].to_int() - 1;
            if (v0 < 0 || v1 < 0 || v2 < 0 || v3 < 0) continue;
            r_mesh.add_tetrahedron(v0, v1, v2, v3, 0);
        }
        r_mesh.precompute_rest_state();
        return OK;
    }

private:
    // Extract vertex index from OBJ face part (e.g., "1/2/3" -> 1)
    static int32_t parse_face_index(const String &p_part) {
        int slash = p_part.find("/");
        if (slash == -1) {
            return p_part.to_int();
        }
        return p_part.left(slash).to_int();
    }
};

} // namespace gaia::mesh

#endif // GAIA_MESH_IO_H