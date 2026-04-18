// godot/xtilemap.hpp

#ifndef XTENSOR_XTILEMAP_HPP
#define XTENSOR_XTILEMAP_HPP

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xexpression.hpp"
#include "../containers/xarray.hpp"
#include "../containers/xtensor.hpp"
#include "../core/xview.hpp"
#include "../core/xfunction.hpp"
#include "../math/xstats.hpp"
#include "../math/xnorm.hpp"
#include "../math/xrandom.hpp"
#include "../math/xinterp.hpp"
#include "../image/ximage_processing.hpp"
#include "xvariant.hpp"
#include "xclassdb.hpp"
#include "xnode.hpp"
#include "xresource.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <map>
#include <unordered_map>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <algorithm>
#include <numeric>
#include <functional>
#include <cmath>
#include <queue>
#include <mutex>
#include <random>

#if XTENSOR_GODOT_CLASSDB_AVAILABLE
    #include <godot_cpp/classes/node2d.hpp>
    #include <godot_cpp/classes/tile_map.hpp>
    #include <godot_cpp/classes/tile_map_layer.hpp>
    #include <godot_cpp/classes/tile_set.hpp>
    #include <godot_cpp/classes/tile_set_source.hpp>
    #include <godot_cpp/classes/tile_set_atlas_source.hpp>
    #include <godot_cpp/classes/tile_data.hpp>
    #include <godot_cpp/classes/navigation_polygon.hpp>
    #include <godot_cpp/classes/occluder_polygon2d.hpp>
    #include <godot_cpp/classes/physics_material.hpp>
    #include <godot_cpp/classes/resource.hpp>
    #include <godot_cpp/classes/world_2d.hpp>
    #include <godot_cpp/classes/navigation_region2d.hpp>
    #include <godot_cpp/variant/utility_functions.hpp>
    #include <godot_cpp/variant/vector2.hpp>
    #include <godot_cpp/variant/vector2i.hpp>
    #include <godot_cpp/variant/rect2i.hpp>
    #include <godot_cpp/variant/packed_vector2_array.hpp>
    #include <godot_cpp/variant/packed_int32_array.hpp>
#endif

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        namespace godot_bridge
        {
            // --------------------------------------------------------------------
            // TileMap Tensor Representation
            // --------------------------------------------------------------------
            // A tilemap can be represented as a 3D tensor: H x W x C
            // where C contains: tile_id, alternative_id, flip_h, flip_v, transpose, layer, etc.
            // For simplicity, we use a 2D tensor of tile IDs and separate flags.

            struct TileDataTensor
            {
                xarray_container<int32_t> tile_ids;      // H x W
                xarray_container<int32_t> alternatives;  // H x W (0 = default)
                xarray_container<uint8_t> flip_h;        // H x W
                xarray_container<uint8_t> flip_v;        // H x W
                xarray_container<uint8_t> transpose;     // H x W
                xarray_container<int32_t> layers;        // H x W (for multi-layer)
                int32_t default_tile = -1;
            };

            // Layer stack: multiple 2D layers
            struct TileMapLayers
            {
                std::vector<xarray_container<int32_t>> layer_data;
                std::vector<std::string> layer_names;
                godot::Vector2i tile_size = godot::Vector2i(16, 16);
            };

            // --------------------------------------------------------------------
            // Procedural TileMap Generation
            // --------------------------------------------------------------------
            class TileMapGenerator
            {
            public:
                // Generate random noise-based tilemap
                static TileDataTensor generate_noise_map(int width, int height,
                                                         const std::vector<int32_t>& tile_palette,
                                                         float scale = 0.1f,
                                                         int seed = 0)
                {
                    TileDataTensor result;
                    result.tile_ids = xarray_container<int32_t>({static_cast<size_t>(height), static_cast<size_t>(width)});
                    result.alternatives = xt::zeros<int32_t>({static_cast<size_t>(height), static_cast<size_t>(width)});
                    result.flip_h = xt::zeros<uint8_t>({static_cast<size_t>(height), static_cast<size_t>(width)});
                    result.flip_v = xt::zeros<uint8_t>({static_cast<size_t>(height), static_cast<size_t>(width)});
                    result.transpose = xt::zeros<uint8_t>({static_cast<size_t>(height), static_cast<size_t>(width)});

                    std::mt19937 rng(static_cast<unsigned int>(seed ? seed : std::random_device{}()));
                    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

                    // Simple Perlin-like noise using multiple octaves
                    for (int y = 0; y < height; ++y)
                    {
                        for (int x = 0; x < width; ++x)
                        {
                            float noise_val = 0.0f;
                            float amp = 1.0f;
                            float freq = scale;
                            for (int oct = 0; oct < 4; ++oct)
                            {
                                float nx = x * freq;
                                float ny = y * freq;
                                noise_val += amp * (std::sin(nx * 1.3f + ny * 0.7f) * std::cos(ny * 1.1f - nx * 0.3f));
                                amp *= 0.5f;
                                freq *= 2.0f;
                            }
                            noise_val = (noise_val + 1.0f) * 0.5f; // normalize to 0-1
                            size_t idx = static_cast<size_t>(noise_val * (tile_palette.size() - 1));
                            idx = std::min(idx, tile_palette.size() - 1);
                            result.tile_ids(y, x) = tile_palette[idx];
                        }
                    }
                    return result;
                }

                // Generate using cellular automata (cave generation)
                static TileDataTensor generate_cellular(int width, int height,
                                                        int32_t wall_tile, int32_t floor_tile,
                                                        float fill_prob = 0.45f,
                                                        int iterations = 4,
                                                        int seed = 0)
                {
                    TileDataTensor result;
                    result.tile_ids = xarray_container<int32_t>({static_cast<size_t>(height), static_cast<size_t>(width)});
                    result.alternatives = xt::zeros<int32_t>({static_cast<size_t>(height), static_cast<size_t>(width)});
                    result.flip_h = xt::zeros<uint8_t>({static_cast<size_t>(height), static_cast<size_t>(width)});
                    result.flip_v = xt::zeros<uint8_t>({static_cast<size_t>(height), static_cast<size_t>(width)});
                    result.transpose = xt::zeros<uint8_t>({static_cast<size_t>(height), static_cast<size_t>(width)});

                    std::mt19937 rng(static_cast<unsigned int>(seed ? seed : std::random_device{}()));
                    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

                    // Initialize random
                    for (int y = 0; y < height; ++y)
                        for (int x = 0; x < width; ++x)
                            result.tile_ids(y, x) = (dist(rng) < fill_prob) ? wall_tile : floor_tile;

                    // Cellular automata steps
                    for (int iter = 0; iter < iterations; ++iter)
                    {
                        auto next = result.tile_ids;
                        for (int y = 1; y < height - 1; ++y)
                        {
                            for (int x = 1; x < width - 1; ++x)
                            {
                                int wall_count = 0;
                                for (int dy = -1; dy <= 1; ++dy)
                                    for (int dx = -1; dx <= 1; ++dx)
                                        if (result.tile_ids(y+dy, x+dx) == wall_tile)
                                            ++wall_count;
                                if (result.tile_ids(y, x) == wall_tile)
                                    next(y, x) = (wall_count >= 4) ? wall_tile : floor_tile;
                                else
                                    next(y, x) = (wall_count >= 5) ? wall_tile : floor_tile;
                            }
                        }
                        result.tile_ids = next;
                    }
                    return result;
                }

                // Generate using Wang tiles / marching squares from a continuous function
                static TileDataTensor generate_from_sdf(const xarray_container<float>& sdf,
                                                        const std::vector<int32_t>& tile_mapping,
                                                        float iso_level = 0.0f)
                {
                    TileDataTensor result;
                    size_t h = sdf.shape()[0];
                    size_t w = sdf.shape()[1];
                    result.tile_ids = xarray_container<int32_t>({h, w});
                    result.alternatives = xt::zeros<int32_t>({h, w});
                    result.flip_h = xt::zeros<uint8_t>({h, w});
                    result.flip_v = xt::zeros<uint8_t>({h, w});
                    result.transpose = xt::zeros<uint8_t>({h, w});

                    for (size_t y = 0; y < h; ++y)
                    {
                        for (size_t x = 0; x < w; ++x)
                        {
                            float val = sdf(y, x);
                            // Map SDF value to tile index
                            int idx = static_cast<int>((val - iso_level) * 10.0f + tile_mapping.size() / 2);
                            idx = std::clamp(idx, 0, static_cast<int>(tile_mapping.size()) - 1);
                            result.tile_ids(y, x) = tile_mapping[idx];
                        }
                    }
                    return result;
                }

                // Generate using wave function collapse (simplified)
                static TileDataTensor generate_wfc(int width, int height,
                                                   const std::vector<std::vector<int32_t>>& adjacency_rules,
                                                   int seed = 0)
                {
                    // Placeholder: wave function collapse algorithm
                    TileDataTensor result;
                    result.tile_ids = xarray_container<int32_t>({static_cast<size_t>(height), static_cast<size_t>(width)});
                    result.alternatives = xt::zeros<int32_t>({static_cast<size_t>(height), static_cast<size_t>(width)});
                    result.flip_h = xt::zeros<uint8_t>({static_cast<size_t>(height), static_cast<size_t>(width)});
                    result.flip_v = xt::zeros<uint8_t>({static_cast<size_t>(height), static_cast<size_t>(width)});
                    result.transpose = xt::zeros<uint8_t>({static_cast<size_t>(height), static_cast<size_t>(width)});
                    return result;
                }
            };

            // --------------------------------------------------------------------
            // XTileMap - Godot node for tensor-based tilemap operations
            // --------------------------------------------------------------------
#if XTENSOR_GODOT_CLASSDB_AVAILABLE
            class XTileMap : public godot::TileMap
            {
                GDCLASS(XTileMap, godot::TileMap)

            private:
                godot::Ref<XTensorNode> m_tile_data;          // H x W (or H x W x C) tensor
                godot::Ref<XTensorNode> m_collision_mask;     // H x W collision flags
                godot::Ref<XTensorNode> m_navigation_mask;    // H x W navigation flags
                godot::String m_tile_set_path;
                int m_default_layer = 0;
                bool m_auto_update = true;
                godot::Vector2i m_map_size = godot::Vector2i(64, 64);
                godot::Ref<godot::TileSet> m_tile_set;

            protected:
                static void _bind_methods()
                {
                    godot::ClassDB::bind_method(godot::D_METHOD("set_tile_data", "tensor"), &XTileMap::set_tile_data);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_tile_data"), &XTileMap::get_tile_data);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_collision_mask", "tensor"), &XTileMap::set_collision_mask);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_collision_mask"), &XTileMap::get_collision_mask);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_navigation_mask", "tensor"), &XTileMap::set_navigation_mask);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_navigation_mask"), &XTileMap::get_navigation_mask);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_tile_set_path", "path"), &XTileMap::set_tile_set_path);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_tile_set_path"), &XTileMap::get_tile_set_path);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_auto_update", "enabled"), &XTileMap::set_auto_update);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_auto_update"), &XTileMap::get_auto_update);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_map_size", "size"), &XTileMap::set_map_size);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_map_size"), &XTileMap::get_map_size);

                    godot::ClassDB::bind_method(godot::D_METHOD("build_from_tensor"), &XTileMap::build_from_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("extract_to_tensor"), &XTileMap::extract_to_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("generate_noise", "width", "height", "tile_palette", "scale", "seed"), &XTileMap::generate_noise, godot::DEFVAL(0.1f), godot::DEFVAL(0));
                    godot::ClassDB::bind_method(godot::D_METHOD("generate_cellular", "width", "height", "wall_tile", "floor_tile", "fill_prob", "iterations", "seed"), &XTileMap::generate_cellular, godot::DEFVAL(0.45f), godot::DEFVAL(4), godot::DEFVAL(0));
                    godot::ClassDB::bind_method(godot::D_METHOD("set_tiles_batch", "positions", "tile_ids"), &XTileMap::set_tiles_batch);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_tiles_batch", "rect"), &XTileMap::get_tiles_batch);
                    godot::ClassDB::bind_method(godot::D_METHOD("clear_layer", "layer"), &XTileMap::clear_layer, godot::DEFVAL(0));
                    godot::ClassDB::bind_method(godot::D_METHOD("flood_fill", "start_pos", "tile_id"), &XTileMap::flood_fill);
                    godot::ClassDB::bind_method(godot::D_METHOD("replace_tiles", "old_id", "new_id"), &XTileMap::replace_tiles);
                    godot::ClassDB::bind_method(godot::D_METHOD("export_to_image", "layer"), &XTileMap::export_to_image, godot::DEFVAL(0));
                    godot::ClassDB::bind_method(godot::D_METHOD("import_from_image", "image", "color_to_tile"), &XTileMap::import_from_image);

                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::OBJECT, "tile_data", godot::PROPERTY_HINT_RESOURCE_TYPE, "XTensorResource"), "set_tile_data", "get_tile_data");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::OBJECT, "collision_mask", godot::PROPERTY_HINT_RESOURCE_TYPE, "XTensorResource"), "set_collision_mask", "get_collision_mask");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::OBJECT, "navigation_mask", godot::PROPERTY_HINT_RESOURCE_TYPE, "XTensorResource"), "set_navigation_mask", "get_navigation_mask");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::STRING, "tile_set_path", godot::PROPERTY_HINT_FILE, "*.tres"), "set_tile_set_path", "get_tile_set_path");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::BOOL, "auto_update"), "set_auto_update", "get_auto_update");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::VECTOR2I, "map_size"), "set_map_size", "get_map_size");

                    ADD_SIGNAL(godot::MethodInfo("map_built"));
                    ADD_SIGNAL(godot::MethodInfo("tiles_updated", godot::PropertyInfo(godot::Variant::RECT2I, "region")));
                }

            public:
                XTileMap() {}

                void _ready() override
                {
                    if (!m_tile_set_path.is_empty())
                    {
                        m_tile_set = godot::ResourceLoader::get_singleton()->load(m_tile_set_path);
                        if (m_tile_set.is_valid())
                            set_tile_set(m_tile_set);
                    }
                    if (m_auto_update && m_tile_data.is_valid())
                        build_from_tensor();
                }

                void set_tile_data(const godot::Ref<XTensorNode>& tensor)
                {
                    m_tile_data = tensor;
                    if (m_auto_update) build_from_tensor();
                }
                godot::Ref<XTensorNode> get_tile_data() const { return m_tile_data; }

                void set_collision_mask(const godot::Ref<XTensorNode>& tensor) { m_collision_mask = tensor; }
                godot::Ref<XTensorNode> get_collision_mask() const { return m_collision_mask; }

                void set_navigation_mask(const godot::Ref<XTensorNode>& tensor) { m_navigation_mask = tensor; }
                godot::Ref<XTensorNode> get_navigation_mask() const { return m_navigation_mask; }

                void set_tile_set_path(const godot::String& path) { m_tile_set_path = path; }
                godot::String get_tile_set_path() const { return m_tile_set_path; }

                void set_auto_update(bool enabled) { m_auto_update = enabled; }
                bool get_auto_update() const { return m_auto_update; }

                void set_map_size(const godot::Vector2i& size) { m_map_size = size; }
                godot::Vector2i get_map_size() const { return m_map_size; }

                void build_from_tensor()
                {
                    if (!m_tile_data.is_valid() || !m_tile_data->is_valid())
                    {
                        godot::UtilityFunctions::printerr("XTileMap: tile_data not set or invalid");
                        return;
                    }

                    auto data = m_tile_data->get_tensor_resource()->m_data.to_int_array();
                    if (data.dimension() != 2)
                    {
                        godot::UtilityFunctions::printerr("XTileMap: tile_data must be 2D (H x W)");
                        return;
                    }

                    size_t h = data.shape()[0];
                    size_t w = data.shape()[1];
                    m_map_size = godot::Vector2i(static_cast<int>(w), static_cast<int>(h));

                    clear_layer(m_default_layer);

                    for (size_t y = 0; y < h; ++y)
                    {
                        for (size_t x = 0; x < w; ++x)
                        {
                            int32_t tile_id = data(y, x);
                            if (tile_id >= 0)
                            {
                                set_cell(m_default_layer, godot::Vector2i(static_cast<int>(x), static_cast<int>(y)),
                                         tile_id, godot::Vector2i(), 0);
                            }
                        }
                    }

                    emit_signal("map_built");
                }

                void extract_to_tensor()
                {
                    if (!m_tile_data.is_valid())
                        m_tile_data.instantiate();

                    godot::Rect2i used_rect = get_used_rect();
                    int w = used_rect.size.x;
                    int h = used_rect.size.y;
                    xarray_container<int32_t> data({static_cast<size_t>(h), static_cast<size_t>(w)});

                    for (int y = 0; y < h; ++y)
                    {
                        for (int x = 0; x < w; ++x)
                        {
                            godot::Vector2i pos(x + used_rect.position.x, y + used_rect.position.y);
                            godot::TileData* tile = get_cell_tile_data(m_default_layer, pos);
                            if (tile)
                                data(y, x) = get_cell_source_id(m_default_layer, pos);
                            else
                                data(y, x) = -1;
                        }
                    }

                    m_tile_data->set_data(XVariant::from_xarray(data.cast<double>()).variant());
                    m_map_size = used_rect.size;
                }

                void generate_noise(int width, int height, const godot::PackedInt32Array& palette,
                                    float scale, int seed)
                {
                    std::vector<int32_t> pal;
                    for (int i = 0; i < palette.size(); ++i)
                        pal.push_back(palette[i]);
                    if (pal.empty()) pal = {0, 1, 2};

                    auto gen = TileMapGenerator::generate_noise_map(width, height, pal, scale, seed);
                    m_tile_data.instantiate();
                    m_tile_data->set_data(XVariant::from_xarray(gen.tile_ids.cast<double>()).variant());
                    m_map_size = godot::Vector2i(width, height);
                    if (m_auto_update) build_from_tensor();
                }

                void generate_cellular(int width, int height, int32_t wall_tile, int32_t floor_tile,
                                       float fill_prob, int iterations, int seed)
                {
                    auto gen = TileMapGenerator::generate_cellular(width, height, wall_tile, floor_tile,
                                                                    fill_prob, iterations, seed);
                    m_tile_data.instantiate();
                    m_tile_data->set_data(XVariant::from_xarray(gen.tile_ids.cast<double>()).variant());
                    m_map_size = godot::Vector2i(width, height);
                    if (m_auto_update) build_from_tensor();
                }

                void set_tiles_batch(const godot::PackedVector2Array& positions, const godot::PackedInt32Array& tile_ids)
                {
                    size_t n = std::min(static_cast<size_t>(positions.size()), static_cast<size_t>(tile_ids.size()));
                    for (size_t i = 0; i < n; ++i)
                    {
                        godot::Vector2 pos = positions[i];
                        set_cell(m_default_layer, godot::Vector2i(static_cast<int>(pos.x), static_cast<int>(pos.y)),
                                 tile_ids[i], godot::Vector2i(), 0);
                    }
                    emit_signal("tiles_updated", godot::Rect2i());
                }

                godot::Dictionary get_tiles_batch(const godot::Rect2i& rect) const
                {
                    godot::Dictionary result;
                    godot::PackedVector2Array positions;
                    godot::PackedInt32Array ids;

                    for (int y = rect.position.y; y < rect.position.y + rect.size.y; ++y)
                    {
                        for (int x = rect.position.x; x < rect.position.x + rect.size.x; ++x)
                        {
                            godot::Vector2i pos(x, y);
                            int32_t id = get_cell_source_id(m_default_layer, pos);
                            if (id != -1)
                            {
                                positions.append(godot::Vector2(x, y));
                                ids.append(id);
                            }
                        }
                    }
                    result["positions"] = positions;
                    result["ids"] = ids;
                    return result;
                }

                void clear_layer(int layer)
                {
                    clear_layer(layer);
                }

                void flood_fill(const godot::Vector2i& start_pos, int32_t tile_id)
                {
                    if (!is_valid_cell(start_pos)) return;
                    int32_t target_id = get_cell_source_id(m_default_layer, start_pos);
                    if (target_id == tile_id) return;

                    std::queue<godot::Vector2i> q;
                    q.push(start_pos);
                    std::unordered_set<int64_t> visited;
                    visited.insert(start_pos.x | (static_cast<int64_t>(start_pos.y) << 32));

                    const int dx[4] = {1, 0, -1, 0};
                    const int dy[4] = {0, 1, 0, -1};

                    while (!q.empty())
                    {
                        godot::Vector2i p = q.front(); q.pop();
                        set_cell(m_default_layer, p, tile_id, godot::Vector2i(), 0);

                        for (int d = 0; d < 4; ++d)
                        {
                            godot::Vector2i np(p.x + dx[d], p.y + dy[d]);
                            int64_t key = np.x | (static_cast<int64_t>(np.y) << 32);
                            if (visited.count(key)) continue;
                            if (!is_valid_cell(np)) continue;
                            if (get_cell_source_id(m_default_layer, np) != target_id) continue;
                            visited.insert(key);
                            q.push(np);
                        }
                    }
                }

                void replace_tiles(int32_t old_id, int32_t new_id)
                {
                    godot::Rect2i used = get_used_rect();
                    for (int y = used.position.y; y < used.position.y + used.size.y; ++y)
                    {
                        for (int x = used.position.x; x < used.position.x + used.size.x; ++x)
                        {
                            godot::Vector2i pos(x, y);
                            if (get_cell_source_id(m_default_layer, pos) == old_id)
                                set_cell(m_default_layer, pos, new_id, godot::Vector2i(), 0);
                        }
                    }
                }

                godot::Ref<godot::Image> export_to_image(int layer) const
                {
                    godot::Rect2i used = get_used_rect();
                    godot::Ref<godot::Image> img;
                    img.instantiate();
                    img->create(used.size.x, used.size.y, false, godot::Image::FORMAT_RGB8);

                    // Create a color mapping from tile IDs
                    std::unordered_map<int32_t, godot::Color> color_map;
                    int hue_step = 360 / 16;
                    for (int i = 0; i < 16; ++i)
                    {
                        color_map[i] = godot::Color::from_hsv(i * hue_step / 360.0f, 0.8f, 0.9f);
                    }

                    for (int y = 0; y < used.size.y; ++y)
                    {
                        for (int x = 0; x < used.size.x; ++x)
                        {
                            godot::Vector2i pos(x + used.position.x, y + used.position.y);
                            int32_t id = get_cell_source_id(layer, pos);
                            godot::Color col = (id >= 0) ? color_map[id % 16] : godot::Color(0, 0, 0);
                            img->set_pixel(x, y, col);
                        }
                    }
                    return img;
                }

                void import_from_image(const godot::Ref<godot::Image>& image, const godot::Dictionary& color_to_tile)
                {
                    if (!image.is_valid()) return;
                    int w = image->get_width();
                    int h = image->get_height();
                    m_map_size = godot::Vector2i(w, h);
                    clear_layer(m_default_layer);

                    xarray_container<int32_t> data({static_cast<size_t>(h), static_cast<size_t>(w)});

                    for (int y = 0; y < h; ++y)
                    {
                        for (int x = 0; x < w; ++x)
                        {
                            godot::Color col = image->get_pixel(x, y);
                            int32_t tile_id = -1;
                            // Find closest color in mapping
                            float min_dist = 1e10f;
                            godot::Array keys = color_to_tile.keys();
                            for (int i = 0; i < keys.size(); ++i)
                            {
                                godot::Color key_col = keys[i];
                                float dr = col.r - key_col.r;
                                float dg = col.g - key_col.g;
                                float db = col.b - key_col.b;
                                float dist = dr*dr + dg*dg + db*db;
                                if (dist < min_dist)
                                {
                                    min_dist = dist;
                                    tile_id = color_to_tile[key_col];
                                }
                            }
                            data(y, x) = tile_id;
                            if (tile_id >= 0)
                                set_cell(m_default_layer, godot::Vector2i(x, y), tile_id, godot::Vector2i(), 0);
                        }
                    }

                    if (!m_tile_data.is_valid()) m_tile_data.instantiate();
                    m_tile_data->set_data(XVariant::from_xarray(data.cast<double>()).variant());
                }

            private:
                bool is_valid_cell(const godot::Vector2i& pos) const
                {
                    return pos.x >= 0 && pos.x < m_map_size.x && pos.y >= 0 && pos.y < m_map_size.y;
                }
            };

            // --------------------------------------------------------------------
            // XTileSetResource - Tensor-based tileset definition
            // --------------------------------------------------------------------
            class XTileSetResource : public godot::Resource
            {
                GDCLASS(XTileSetResource, godot::Resource)

            private:
                godot::Ref<XTensorNode> m_tile_properties;  // N x P: properties for each tile
                godot::Ref<XTensorNode> m_adjacency_matrix; // N x N: which tiles can be adjacent
                godot::PackedStringArray m_property_names;
                godot::String m_texture_path;
                godot::Vector2i m_tile_size = godot::Vector2i(16, 16);
                int m_tiles_per_row = 8;

            protected:
                static void _bind_methods()
                {
                    godot::ClassDB::bind_method(godot::D_METHOD("set_tile_properties", "tensor"), &XTileSetResource::set_tile_properties);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_tile_properties"), &XTileSetResource::get_tile_properties);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_adjacency_matrix", "tensor"), &XTileSetResource::set_adjacency_matrix);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_adjacency_matrix"), &XTileSetResource::get_adjacency_matrix);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_property_names", "names"), &XTileSetResource::set_property_names);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_property_names"), &XTileSetResource::get_property_names);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_texture_path", "path"), &XTileSetResource::set_texture_path);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_texture_path"), &XTileSetResource::get_texture_path);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_tile_size", "size"), &XTileSetResource::set_tile_size);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_tile_size"), &XTileSetResource::get_tile_size);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_tile_count"), &XTileSetResource::get_tile_count);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_tile_texture_region", "tile_id"), &XTileSetResource::get_tile_texture_region);

                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::OBJECT, "tile_properties", godot::PROPERTY_HINT_RESOURCE_TYPE, "XTensorResource"), "set_tile_properties", "get_tile_properties");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::OBJECT, "adjacency_matrix", godot::PROPERTY_HINT_RESOURCE_TYPE, "XTensorResource"), "set_adjacency_matrix", "get_adjacency_matrix");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::PACKED_STRING_ARRAY, "property_names"), "set_property_names", "get_property_names");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::STRING, "texture_path", godot::PROPERTY_HINT_FILE, "*.png"), "set_texture_path", "get_texture_path");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::VECTOR2I, "tile_size"), "set_tile_size", "get_tile_size");
                }

            public:
                void set_tile_properties(const godot::Ref<XTensorNode>& tensor) { m_tile_properties = tensor; }
                godot::Ref<XTensorNode> get_tile_properties() const { return m_tile_properties; }

                void set_adjacency_matrix(const godot::Ref<XTensorNode>& tensor) { m_adjacency_matrix = tensor; }
                godot::Ref<XTensorNode> get_adjacency_matrix() const { return m_adjacency_matrix; }

                void set_property_names(const godot::PackedStringArray& names) { m_property_names = names; }
                godot::PackedStringArray get_property_names() const { return m_property_names; }

                void set_texture_path(const godot::String& path) { m_texture_path = path; }
                godot::String get_texture_path() const { return m_texture_path; }

                void set_tile_size(const godot::Vector2i& size) { m_tile_size = size; }
                godot::Vector2i get_tile_size() const { return m_tile_size; }

                int get_tile_count() const
                {
                    if (m_tile_properties.is_valid())
                        return static_cast<int>(m_tile_properties->get_tensor_resource()->m_data.shape()[0]);
                    return 0;
                }

                godot::Rect2i get_tile_texture_region(int tile_id) const
                {
                    int row = tile_id / m_tiles_per_row;
                    int col = tile_id % m_tiles_per_row;
                    return godot::Rect2i(col * m_tile_size.x, row * m_tile_size.y, m_tile_size.x, m_tile_size.y);
                }
            };
#endif // XTENSOR_GODOT_CLASSDB_AVAILABLE

            // --------------------------------------------------------------------
            // Registration
            // --------------------------------------------------------------------
            class XTileMapRegistry
            {
            public:
                static void register_classes()
                {
#if XTENSOR_GODOT_CLASSDB_AVAILABLE
                    godot::ClassDB::register_class<XTileMap>();
                    godot::ClassDB::register_class<XTileSetResource>();
#endif
                }
            };

        } // namespace godot_bridge

        using godot_bridge::TileDataTensor;
        using godot_bridge::TileMapGenerator;
        using godot_bridge::XTileMap;
        using godot_bridge::XTileSetResource;
        using godot_bridge::XTileMapRegistry;

    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XTILEMAP_HPP

// godot/xtilemap.hpp