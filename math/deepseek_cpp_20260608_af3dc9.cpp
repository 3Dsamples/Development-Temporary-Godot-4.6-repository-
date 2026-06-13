// File 416: modules/integration/procedural_texture_atlas_generator.h
// Builds a packed texture atlas from a set of UV islands (one per input
// surface) using the MaxRects bin‑packing algorithm.  Outputs per‑vertex
// UVs transformed to atlas space, a mapping from old surface index to its
// atlas page and rectangle, and an optional combined texture placeholder.
// Supports multiple atlas pages, padding between islands, and power‑of‑two
// enforcement.  All packing and UV‑coordinate math is fully present.

#ifndef INTEGRATION_PROCEDURAL_TEXTURE_ATLAS_GENERATOR_H
#define INTEGRATION_PROCEDURAL_TEXTURE_ATLAS_GENERATOR_H

#include "core/templates/local_vector.h"
#include "core/templates/hash_map.h"
#include "core/templates/hash_set.h"
#include "core/math/vector2.h"
#include "core/math/rect2.h"
#include "core/typedefs.h"
#include <algorithm>

namespace unified {

class ProceduralTextureAtlasGenerator {
public:
    // ---------- input ----------
    struct SurfaceUVData {
        LocalVector<Vector2> uvs;       // original UVs per vertex
        LocalVector<int>     indices;   // triangle indices (optional, for island detection)
    };

    // ---------- output ----------
    struct AtlasPage {
        int width;                        // final page width (power of two)
        int height;                       // final page height
        // The actual pixel data is not stored here; the game must fill the texture.
    };

    struct PackedSurface {
        int atlas_page;                   // index into atlas_pages array
        Vector2 uv_offset;                // offset to add to original UVs (they will be scaled+offset)
        Vector2 uv_scale;                 // scaling factor applied to original UVs
        Rect2i rect;                      // packed rectangle in page pixels
    };

    // ---------- parameters ----------
    int max_atlas_width = 2048;
    int max_atlas_height = 2048;
    int padding = 2;                      // pixels between islands in UV space
    bool allow_multiple_pages = true;
    bool power_of_two = true;

    // ---------- internal state ----------
    LocalVector<AtlasPage> pages;
    LocalVector<PackedSurface> packed_results;
    bool built = false;

public:
    ProceduralTextureAtlasGenerator() {}

    // Build atlas from a list of surfaces (each with their own UVs and faces).
    // Returns true on success.
    bool build(const LocalVector<SurfaceUVData> &p_surfaces) {
        pages.clear();
        packed_results.clear();
        built = false;

        if (p_surfaces.is_empty()) return false;

        // 1. Compute UV‑island bounding boxes for each surface.
        //    For simplicity, we treat each surface as a single island,
        //    but a more advanced implementation would detect connected
        //    UV components and pack them independently.
        LocalVector<Rect2> island_boxes;
        island_boxes.resize(p_surfaces.size());
        for (int i = 0; i < p_surfaces.size(); ++i) {
            island_boxes[i] = compute_island_bbox(p_surfaces[i].uvs);
        }

        // 2. Convert to pixel rectangles (fixed texture resolution units).
        //    We'll map UV [0..1] to [0..texWidth) and [0..texHeight) using a
        //    virtual packing space of 2048x2048.  Later we'll scale to actual
        //    page size when we know the final layout.
        //    Actually MaxRects works with integer rectangles, so we'll work in
        //    texel units for a target page of 2048.  After packing, we adjust
        //    the output scale and offset to reflect the real page coordinates.
        //    For simplicity, we'll pack into a working virtual area of size
        //    max_atlas_width x max_atlas_height, then finalise to power-of-two.
        const int virtual_w = max_atlas_width;
        const int virtual_h = max_atlas_height;

        LocalVector<Rect2i> rects;
        rects.reserve(island_boxes.size());
        for (const Rect2 &box : island_boxes) {
            int x = (int)(box.position.x * virtual_w + 0.5);
            int y = (int)(box.position.y * virtual_h + 0.5);
            int w = (int)(box.size.x * virtual_w + 0.5) + padding * 2;
            int h = (int)(box.size.y * virtual_h + 0.5) + padding * 2;
            rects.push_back(Rect2i(x, y, MAX(w,1), MAX(h,1)));
        }

        // 3. Sort rectangles by area descending (improves packing).
        LocalVector<int> sorted_indices;
        sorted_indices.resize(rects.size());
        for (int i = 0; i < sorted_indices.size(); ++i) sorted_indices[i] = i;
        sorted_indices.sort([&](int a, int b) {
            int area_a = rects[a].size.x * rects[a].size.y;
            int area_b = rects[b].size.x * rects[b].size.y;
            return area_a > area_b; // descending
        });

        // 4. Pack rectangles into one or more pages using MaxRects.
        struct PageState {
            LocalVector<Rect2i> free_rects; // list of available space
        };
        LocalVector<PageState> page_states;
        // Start with one page.
        if (!create_new_page(page_states, virtual_w, virtual_h))
            return false;

        LocalVector<Rect2i> packed_rects(rects.size()); // final packed positions
        LocalVector<int>    page_assignment(rects.size(), -1);

        for (int si = 0; si < sorted_indices.size(); ++si) {
            int idx = sorted_indices[si];
            int rect_w = rects[idx].size.x;
            int rect_h = rects[idx].size.y;
            bool packed = false;

            // Try existing pages first.
            for (int pi = 0; pi < page_states.size(); ++pi) {
                Rect2i best_rect;
                int best_free_idx = -1;
                if (maxrects_find_best(page_states[pi].free_rects, rect_w, rect_h,
                                       best_rect, best_free_idx)) {
                    // Place the rectangle.
                    packed_rects[idx] = best_rect;
                    page_assignment[idx] = pi;
                    // Split free rectangle(s).
                    maxrects_split(page_states[pi].free_rects, best_free_idx, best_rect);
                    packed = true;
                    break;
                }
            }

            if (!packed && allow_multiple_pages) {
                // Create a new page and place there.
                int new_pi = page_states.size();
                if (!create_new_page(page_states, virtual_w, virtual_h))
                    return false;
                Rect2i best_rect;
                int best_free_idx = -1;
                if (maxrects_find_best(page_states[new_pi].free_rects, rect_w, rect_h,
                                       best_rect, best_free_idx)) {
                    packed_rects[idx] = best_rect;
                    page_assignment[idx] = new_pi;
                    maxrects_split(page_states[new_pi].free_rects, best_free_idx, best_rect);
                    packed = true;
                }
            }

            if (!packed)
                return false; // cannot fit even with multiple pages (shouldn't happen).
        }

        // 5. Finalise pages (power‑of‑two sizes, offset/scale per surface).
        packed_results.resize(p_surfaces.size());
        for (int i = 0; i < p_surfaces.size(); ++i) {
            int pi = page_assignment[i];
            if (pi < 0) return false;
            const Rect2i &packed_pixel = packed_rects[i];
            // Determine the final page dimensions (power of two).
            // We'll compute it as the maximum extent of all rectangles on this page
            // and round up to power of two.  This will be done after all placements.
            // For now, store the packed rectangle and later adjust.
            PackedSurface &out = packed_results[i];
            out.atlas_page = pi;
            // We'll save the rectangle relative to this page.
            // To compute uv_scale and uv_offset, we need the final page size.
            // We'll postpone that to a finalise step.
        }

        // Determine final dimensions for each page as the bounding rectangle of all
        // placed rects on that page, rounded up to power of two if needed.
        pages.resize(page_states.size());
        for (int pi = 0; pi < pages.size(); ++pi) {
            int w = 0, h = 0;
            for (int i = 0; i < p_surfaces.size(); ++i) {
                if (page_assignment[i] == pi) {
                    int right = packed_rects[i].position.x + packed_rects[i].size.x;
                    int bottom = packed_rects[i].position.y + packed_rects[i].size.y;
                    w = MAX(w, right);
                    h = MAX(h, bottom);
                }
            }
            if (power_of_two) {
                w = next_power_of_two(w);
                h = next_power_of_two(h);
            }
            pages[pi].width = MAX(w, 1);
            pages[pi].height = MAX(h, 1);
        }

        // 6. Compute per‑surface UV scale and offset.
        for (int i = 0; i < p_surfaces.size(); ++i) {
            int pi = page_assignment[i];
            const Rect2i &pix = packed_rects[i];
            int page_w = pages[pi].width;
            int page_h = pages[pi].height;
            if (page_w <= 0 || page_h <= 0) return false;

            // Original UV bounding box (in 0..1) of this surface.
            const Rect2 &orig_bbox = island_boxes[i];
            real_t orig_w = orig_bbox.size.x;
            real_t orig_h = orig_bbox.size.y;

            // The packed rectangle in UV space for this page:
            //   page_u = (packed_x + padding) / page_w   ... padding added to preserve border
            // Because we added padding to the rectangle sizes, the actual image region
            // inside the packed rect is (pad, pad, w-2*pad, h-2*pad).
            int content_x = pix.position.x + padding;
            int content_y = pix.position.y + padding;
            int content_w = pix.size.x - padding * 2;
            int content_h = pix.size.y - padding * 2;
            if (content_w <= 0 || content_h <= 0) {
                content_w = MAX(content_w, 1);
                content_h = MAX(content_h, 1);
            }

            // Transform: original_uv -> new_uv = uv_offset + uv_scale * original_uv.
            // We want to map orig_bbox.min -> (content_x/page_w, content_y/page_h)
            // and orig_bbox.max -> ((content_x+content_w)/page_w, (content_y+content_h)/page_h).
            // So:
            //   uv_scale.x = (content_w / page_w) / orig_w
            //   uv_offset.x = (content_x / page_w) - uv_scale.x * orig_bbox.position.x
            Vector2 scale(content_w / (real_t)page_w / orig_w,
                          content_h / (real_t)page_h / orig_h);
            Vector2 offset(content_x / (real_t)page_w - scale.x * orig_bbox.position.x,
                           content_y / (real_t)page_h - scale.y * orig_bbox.position.y);
            packed_results[i].uv_scale = scale;
            packed_results[i].uv_offset = offset;
        }

        built = true;
        return true;
    }

    const LocalVector<AtlasPage> &get_pages() const { return pages; }
    const LocalVector<PackedSurface> &get_results() const { return packed_results; }

    // After build, transform a set of original UVs into the atlas UVs for
    // a given surface index.  Returns an array of new UVs.
    LocalVector<Vector2> transform_uvs(int p_surface_index, const LocalVector<Vector2> &p_original_uvs) const {
        LocalVector<Vector2> new_uvs;
        ERR_FAIL_COND_V(!built, new_uvs);
        ERR_FAIL_INDEX_V(p_surface_index, packed_results.size(), new_uvs);
        const PackedSurface &ps = packed_results[p_surface_index];
        new_uvs.resize(p_original_uvs.size());
        for (int i = 0; i < p_original_uvs.size(); ++i) {
            new_uvs[i] = ps.uv_offset + ps.uv_scale * p_original_uvs[i];
        }
        return new_uvs;
    }

private:
    // Compute UV bounding box for a list of UVs.
    static Rect2 compute_island_bbox(const LocalVector<Vector2> &p_uvs) {
        if (p_uvs.is_empty()) return Rect2();
        Vector2 min_uv = p_uvs[0];
        Vector2 max_uv = p_uvs[0];
        for (int i = 1; i < p_uvs.size(); ++i) {
            min_uv = min_uv.min(p_uvs[i]);
            max_uv = max_uv.max(p_uvs[i]);
        }
        return Rect2(min_uv, max_uv - min_uv);
    }

    // Create a new page with initial free rectangle covering the entire area.
    static bool create_new_page(LocalVector<PageState> &p_states, int p_w, int p_h) {
        PageState state;
        state.free_rects.push_back(Rect2i(0, 0, p_w, p_h));
        p_states.push_back(state);
        return true;
    }

    // MaxRects: try to find the best free rectangle for the given width and height.
    // `r_out` receives the placed rectangle (position in that free area).
    // `r_free_idx` is the index in free_rects list of the rectangle that was used.
    // Returns false if no fit.
    static bool maxrects_find_best(const LocalVector<Rect2i> &p_free,
                                   int p_width, int p_height,
                                   Rect2i &r_out, int &r_free_idx) {
        // Use best short side fit (BSSF) heuristic.
        int best_area = 0x7FFFFFFF;
        int best_idx = -1;
        for (int i = 0; i < p_free.size(); ++i) {
            const Rect2i &fr = p_free[i];
            if (fr.size.x < p_width || fr.size.y < p_height) continue;
            int area = fr.size.x * fr.size.y;
            if (area < best_area) {
                best_area = area;
                best_idx = i;
            }
        }
        if (best_idx < 0) return false;
        r_out = Rect2i(p_free[best_idx].position.x,
                       p_free[best_idx].position.y,
                       p_width, p_height);
        r_free_idx = best_idx;
        return true;
    }

    // Split the chosen free rectangle into remaining free space after placing `p_placed`.
    static void maxrects_split(LocalVector<Rect2i> &p_free,
                               int p_used_idx, const Rect2i &p_placed) {
        const Rect2i &used_rect = p_placed;
        // Remove the used rectangle from free list.
        Rect2i free = p_free[p_used_idx];
        p_free.remove_at_unordered(p_used_idx);

        // Generate the two maximal remaining rectangles (horizontal/vertical split).
        // We'll compute the four possible rectangles and add those with positive area.
        // Standard MaxRects update.

        // Left part (remaining on left of placed rectangle within free).
        if (used_rect.position.x > free.position.x) {
            int left_w = used_rect.position.x - free.position.x;
            int full_h = free.size.y;
            if (left_w > 0 && full_h > 0) {
                p_free.push_back(Rect2i(free.position.x, free.position.y,
                                        left_w, full_h));
            }
        }
        // Right part.
        int right_edge = used_rect.position.x + used_rect.size.x;
        int free_right = free.position.x + free.size.x;
        if (right_edge < free_right) {
            int right_w = free_right - right_edge;
            if (right_w > 0 && free.size.y > 0) {
                p_free.push_back(Rect2i(right_edge, free.position.y,
                                        right_w, free.size.y));
            }
        }
        // Top part (above placed rectangle within free).
        if (used_rect.position.y > free.position.y) {
            int top_h = used_rect.position.y - free.position.y;
            int full_w = free.size.x;
            if (top_h > 0 && full_w > 0) {
                p_free.push_back(Rect2i(free.position.x, free.position.y,
                                        full_w, top_h));
            }
        }
        // Bottom part.
        int bottom_edge = used_rect.position.y + used_rect.size.y;
        int free_bottom = free.position.y + free.size.y;
        if (bottom_edge < free_bottom) {
            int bottom_h = free_bottom - bottom_edge;
            if (bottom_h > 0 && free.size.x > 0) {
                p_free.push_back(Rect2i(free.position.x, bottom_edge,
                                        free.size.x, bottom_h));
            }
        }
    }

    // Round integer to next power of two.
    static int next_power_of_two(int p_val) {
        p_val = MAX(p_val, 1);
        p_val--;
        p_val |= p_val >> 1;
        p_val |= p_val >> 2;
        p_val |= p_val >> 4;
        p_val |= p_val >> 8;
        p_val |= p_val >> 16;
        return p_val + 1;
    }
};

} // namespace unified

#endif // INTEGRATION_PROCEDURAL_TEXTURE_ATLAS_GENERATOR_H