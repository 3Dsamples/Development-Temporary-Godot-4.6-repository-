// godot/xgui.hpp

#ifndef XTENSOR_XGUI_HPP
#define XTENSOR_XGUI_HPP

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xexpression.hpp"
#include "../containers/xarray.hpp"
#include "../containers/xtensor.hpp"
#include "../core/xview.hpp"
#include "../core/xfunction.hpp"
#include "../math/xstats.hpp"
#include "../math/xnorm.hpp"
#include "../math/xinterp.hpp"
#include "../math/xcolormap.hpp"
#include "../graphics/xgraphics.hpp"
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

#if XTENSOR_GODOT_CLASSDB_AVAILABLE
    #include <godot_cpp/classes/control.hpp>
    #include <godot_cpp/classes/label.hpp>
    #include <godot_cpp/classes/button.hpp>
    #include <godot_cpp/classes/line_edit.hpp>
    #include <godot_cpp/classes/text_edit.hpp>
    #include <godot_cpp/classes/rich_text_label.hpp>
    #include <godot_cpp/classes/progress_bar.hpp>
    #include <godot_cpp/classes/slider.hpp>
    #include <godot_cpp/classes/scroll_bar.hpp>
    #include <godot_cpp/classes/check_box.hpp>
    #include <godot_cpp/classes/check_button.hpp>
    #include <godot_cpp/classes/option_button.hpp>
    #include <godot_cpp/classes/menu_button.hpp>
    #include <godot_cpp/classes/color_picker_button.hpp>
    #include <godot_cpp/classes/color_picker.hpp>
    #include <godot_cpp/classes/popup_menu.hpp>
    #include <godot_cpp/classes/panel.hpp>
    #include <godot_cpp/classes/panel_container.hpp>
    #include <godot_cpp/classes/margin_container.hpp>
    #include <godot_cpp/classes/center_container.hpp>
    #include <godot_cpp/classes/box_container.hpp>
    #include <godot_cpp/classes/h_box_container.hpp>
    #include <godot_cpp/classes/v_box_container.hpp>
    #include <godot_cpp/classes/grid_container.hpp>
    #include <godot_cpp/classes/scroll_container.hpp>
    #include <godot_cpp/classes/tab_container.hpp>
    #include <godot_cpp/classes/split_container.hpp>
    #include <godot_cpp/classes/item_list.hpp>
    #include <godot_cpp/classes/tree.hpp>
    #include <godot_cpp/classes/tree_item.hpp>
    #include <godot_cpp/classes/color_rect.hpp>
    #include <godot_cpp/classes/texture_rect.hpp>
    #include <godot_cpp/classes/nine_patch_rect.hpp>
    #include <godot_cpp/classes/graph_edit.hpp>
    #include <godot_cpp/classes/graph_node.hpp>
    #include <godot_cpp/classes/viewport.hpp>
    #include <godot_cpp/classes/sub_viewport.hpp>
    #include <godot_cpp/classes/window.hpp>
    #include <godot_cpp/classes/file_dialog.hpp>
    #include <godot_cpp/classes/confirmation_dialog.hpp>
    #include <godot_cpp/classes/accept_dialog.hpp>
    #include <godot_cpp/classes/theme.hpp>
    #include <godot_cpp/classes/font.hpp>
    #include <godot_cpp/classes/font_file.hpp>
    #include <godot_cpp/classes/style_box.hpp>
    #include <godot_cpp/classes/style_box_flat.hpp>
    #include <godot_cpp/classes/style_box_texture.hpp>
    #include <godot_cpp/classes/input_event.hpp>
    #include <godot_cpp/variant/utility_functions.hpp>
    #include <godot_cpp/variant/color.hpp>
    #include <godot_cpp/variant/vector2.hpp>
    #include <godot_cpp/variant/rect2.hpp>
#endif

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        namespace godot_bridge
        {
            // --------------------------------------------------------------------
            // GUI Property Binding: map tensor data to control properties
            // --------------------------------------------------------------------
            struct GUIPropertyBinding
            {
                godot::NodePath control_path;
                godot::StringName property;
                size_t tensor_index;        // Which row/element in the tensor
                size_t component_offset;    // Offset within the tensor row (for multi-component props)
                float scale = 1.0f;
                float bias = 0.0f;
                bool enabled = true;
            };

            class XGUIBindingManager
            {
            public:
                std::vector<GUIPropertyBinding> bindings;
                xarray_container<float> data; // N x M tensor (N entities, M properties)
                
                void apply_bindings(godot::Node* root)
                {
                    if (!root) return;
                    for (const auto& b : bindings)
                    {
                        if (!b.enabled) continue;
                        godot::Node* ctrl = root->get_node_or_null(b.control_path);
                        if (!ctrl) continue;
                        if (b.tensor_index >= data.shape()[0]) continue;
                        
                        float val = data(b.tensor_index, b.component_offset) * b.scale + b.bias;
                        ctrl->set(b.property, val);
                    }
                }
                
                void update_from_controls(godot::Node* root)
                {
                    if (!root) return;
                    for (const auto& b : bindings)
                    {
                        if (!b.enabled) continue;
                        godot::Node* ctrl = root->get_node_or_null(b.control_path);
                        if (!ctrl) continue;
                        if (b.tensor_index >= data.shape()[0]) continue;
                        
                        godot::Variant var = ctrl->get(b.property);
                        float val = static_cast<float>(var);
                        data(b.tensor_index, b.component_offset) = (val - b.bias) / b.scale;
                    }
                }
            };

            // --------------------------------------------------------------------
            // XTensorPlot - Canvas for plotting tensor data
            // --------------------------------------------------------------------
#if XTENSOR_GODOT_CLASSDB_AVAILABLE
            class XTensorPlot : public godot::Control
            {
                GDCLASS(XTensorPlot, godot::Control)

            public:
                enum PlotType
                {
                    PLOT_LINE = 0,
                    PLOT_SCATTER = 1,
                    PLOT_BAR = 2,
                    PLOT_HISTOGRAM = 3,
                    PLOT_IMAGE = 4,
                    PLOT_HEATMAP = 5
                };

            private:
                godot::Ref<XTensorNode> m_data_tensor;
                PlotType m_plot_type = PLOT_LINE;
                godot::Color m_line_color = godot::Color(0.2f, 0.6f, 1.0f, 1.0f);
                godot::Color m_fill_color = godot::Color(0.2f, 0.6f, 1.0f, 0.3f);
                float m_line_width = 2.0f;
                float m_point_size = 4.0f;
                bool m_show_points = true;
                godot::Vector2 m_margins = godot::Vector2(40, 20);
                godot::Rect2 m_data_rect;
                bool m_auto_update = true;
                bool m_dirty = true;
                
                // Cached rendering
                godot::Ref<godot::ImageTexture> m_canvas_texture;
                godot::Ref<godot::Image> m_canvas_image;
                
                // Axis limits
                bool m_auto_x_range = true;
                bool m_auto_y_range = true;
                godot::Vector2 m_x_range = godot::Vector2(0, 1);
                godot::Vector2 m_y_range = godot::Vector2(0, 1);
                
                // Colormap for heatmap/image
                godot::String m_colormap = "viridis";
                std::shared_ptr<graphics::Colormap> m_cmap;

            protected:
                static void _bind_methods()
                {
                    godot::ClassDB::bind_method(godot::D_METHOD("set_data_tensor", "tensor"), &XTensorPlot::set_data_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_data_tensor"), &XTensorPlot::get_data_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_plot_type", "type"), &XTensorPlot::set_plot_type);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_plot_type"), &XTensorPlot::get_plot_type);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_line_color", "color"), &XTensorPlot::set_line_color);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_line_color"), &XTensorPlot::get_line_color);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_fill_color", "color"), &XTensorPlot::set_fill_color);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_fill_color"), &XTensorPlot::get_fill_color);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_line_width", "width"), &XTensorPlot::set_line_width);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_line_width"), &XTensorPlot::get_line_width);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_show_points", "show"), &XTensorPlot::set_show_points);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_show_points"), &XTensorPlot::get_show_points);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_auto_update", "enabled"), &XTensorPlot::set_auto_update);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_auto_update"), &XTensorPlot::get_auto_update);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_x_range", "min", "max"), &XTensorPlot::set_x_range);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_y_range", "min", "max"), &XTensorPlot::set_y_range);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_colormap", "name"), &XTensorPlot::set_colormap);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_colormap"), &XTensorPlot::get_colormap);
                    
                    godot::ClassDB::bind_method(godot::D_METHOD("refresh"), &XTensorPlot::refresh);
                    godot::ClassDB::bind_method(godot::D_METHOD("clear"), &XTensorPlot::clear);
                    
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::OBJECT, "data_tensor", godot::PROPERTY_HINT_RESOURCE_TYPE, "XTensorResource"), "set_data_tensor", "get_data_tensor");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::INT, "plot_type", godot::PROPERTY_HINT_ENUM, "Line,Scatter,Bar,Histogram,Image,Heatmap"), "set_plot_type", "get_plot_type");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::COLOR, "line_color"), "set_line_color", "get_line_color");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::COLOR, "fill_color"), "set_fill_color", "get_fill_color");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::FLOAT, "line_width"), "set_line_width", "get_line_width");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::BOOL, "show_points"), "set_show_points", "get_show_points");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::BOOL, "auto_update"), "set_auto_update", "get_auto_update");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::STRING, "colormap"), "set_colormap", "get_colormap");
                    
                    BIND_ENUM_CONSTANT(PLOT_LINE);
                    BIND_ENUM_CONSTANT(PLOT_SCATTER);
                    BIND_ENUM_CONSTANT(PLOT_BAR);
                    BIND_ENUM_CONSTANT(PLOT_HISTOGRAM);
                    BIND_ENUM_CONSTANT(PLOT_IMAGE);
                    BIND_ENUM_CONSTANT(PLOT_HEATMAP);
                }

            public:
                XTensorPlot()
                {
                    m_cmap = graphics::get_colormap("viridis");
                    set_clip_contents(true);
                }
                
                void _ready() override
                {
                    refresh();
                }
                
                void _draw() override
                {
                    if (m_dirty) render_plot();
                    if (m_canvas_texture.is_valid())
                    {
                        draw_texture(m_canvas_texture, godot::Vector2(0, 0));
                    }
                }
                
                void _notification(int what)
                {
                    if (what == NOTIFICATION_RESIZED)
                    {
                        m_dirty = true;
                        queue_redraw();
                    }
                }

                void set_data_tensor(const godot::Ref<XTensorNode>& tensor)
                {
                    m_data_tensor = tensor;
                    m_dirty = true;
                    if (m_auto_update) queue_redraw();
                }
                
                godot::Ref<XTensorNode> get_data_tensor() const { return m_data_tensor; }
                
                void set_plot_type(PlotType type) { m_plot_type = type; m_dirty = true; }
                PlotType get_plot_type() const { return m_plot_type; }
                
                void set_line_color(const godot::Color& c) { m_line_color = c; m_dirty = true; }
                godot::Color get_line_color() const { return m_line_color; }
                
                void set_fill_color(const godot::Color& c) { m_fill_color = c; m_dirty = true; }
                godot::Color get_fill_color() const { return m_fill_color; }
                
                void set_line_width(float w) { m_line_width = w; m_dirty = true; }
                float get_line_width() const { return m_line_width; }
                
                void set_show_points(bool show) { m_show_points = show; m_dirty = true; }
                bool get_show_points() const { return m_show_points; }
                
                void set_auto_update(bool enabled) { m_auto_update = enabled; }
                bool get_auto_update() const { return m_auto_update; }
                
                void set_x_range(float min, float max) { m_x_range = godot::Vector2(min, max); m_auto_x_range = false; m_dirty = true; }
                void set_y_range(float min, float max) { m_y_range = godot::Vector2(min, max); m_auto_y_range = false; m_dirty = true; }
                
                void set_colormap(const godot::String& name)
                {
                    m_colormap = name;
                    m_cmap = graphics::get_colormap(name.utf8().get_data());
                    m_dirty = true;
                }
                
                godot::String get_colormap() const { return m_colormap; }
                
                void refresh()
                {
                    m_dirty = true;
                    queue_redraw();
                }
                
                void clear()
                {
                    m_data_tensor.unref();
                    m_canvas_texture.unref();
                    m_canvas_image.unref();
                    m_dirty = true;
                    queue_redraw();
                }

            private:
                void render_plot()
                {
                    godot::Vector2 size = get_size();
                    if (size.x <= 0 || size.y <= 0) return;
                    
                    m_data_rect = godot::Rect2(m_margins, size - m_margins * 2.0f);
                    if (m_data_rect.size.x <= 0 || m_data_rect.size.y <= 0) return;
                    
                    // Create image to render into
                    m_canvas_image.instantiate();
                    m_canvas_image->create(size.x, size.y, false, godot::Image::FORMAT_RGBA8);
                    m_canvas_image->fill(godot::Color(0, 0, 0, 0).to_rgba32());
                    
                    // Draw axes
                    draw_axes();
                    
                    // Draw data based on type
                    if (m_data_tensor.is_valid() && m_data_tensor->is_valid())
                    {
                        auto data = m_data_tensor->get_tensor_resource()->m_data.to_double_array();
                        switch (m_plot_type)
                        {
                            case PLOT_LINE:
                            case PLOT_SCATTER:
                                draw_line_plot(data);
                                break;
                            case PLOT_BAR:
                                draw_bar_plot(data);
                                break;
                            case PLOT_HISTOGRAM:
                                draw_histogram(data);
                                break;
                            case PLOT_IMAGE:
                            case PLOT_HEATMAP:
                                draw_heatmap(data);
                                break;
                            default:
                                break;
                        }
                    }
                    
                    m_canvas_texture = godot::ImageTexture::create_from_image(m_canvas_image);
                    m_dirty = false;
                }
                
                void draw_axes()
                {
                    // Draw axes lines
                    godot::Vector2 origin = m_data_rect.position + godot::Vector2(0, m_data_rect.size.y);
                    godot::Vector2 x_end = origin + godot::Vector2(m_data_rect.size.x, 0);
                    godot::Vector2 y_end = origin - godot::Vector2(0, m_data_rect.size.y);
                    
                    draw_line_canvas(origin, x_end, godot::Color(0.5, 0.5, 0.5), 1.0f);
                    draw_line_canvas(origin, y_end, godot::Color(0.5, 0.5, 0.5), 1.0f);
                    
                    // Draw tick labels (simplified)
                }
                
                void draw_line_plot(const xarray_container<double>& data)
                {
                    if (data.size() == 0) return;
                    
                    // Determine ranges
                    godot::Vector2 x_range, y_range;
                    compute_ranges(data, x_range, y_range);
                    
                    size_t n = data.shape()[0];
                    std::vector<godot::Vector2> points;
                    points.reserve(n);
                    
                    for (size_t i = 0; i < n; ++i)
                    {
                        float x = static_cast<float>(i) / (n - 1);
                        float y = static_cast<float>((data(i) - y_range.x) / (y_range.y - y_range.x + 1e-6f));
                        godot::Vector2 pt = m_data_rect.position + godot::Vector2(x * m_data_rect.size.x, m_data_rect.size.y - y * m_data_rect.size.y);
                        points.push_back(pt);
                    }
                    
                    // Draw line
                    for (size_t i = 1; i < points.size(); ++i)
                    {
                        draw_line_canvas(points[i-1], points[i], m_line_color, m_line_width);
                    }
                    
                    // Draw points
                    if (m_show_points)
                    {
                        for (const auto& pt : points)
                        {
                            draw_circle_canvas(pt, m_point_size, m_line_color);
                        }
                    }
                    
                    // Fill under curve if fill color alpha > 0
                    if (m_fill_color.a > 0)
                    {
                        std::vector<godot::Vector2> fill_poly = points;
                        fill_poly.push_back(godot::Vector2(points.back().x, m_data_rect.position.y + m_data_rect.size.y));
                        fill_poly.push_back(godot::Vector2(points.front().x, m_data_rect.position.y + m_data_rect.size.y));
                        draw_polygon_canvas(fill_poly, m_fill_color);
                    }
                }
                
                void draw_bar_plot(const xarray_container<double>& data)
                {
                    size_t n = data.shape()[0];
                    float bar_width = m_data_rect.size.x / (n * 1.5f);
                    
                    godot::Vector2 x_range, y_range;
                    compute_ranges(data, x_range, y_range);
                    
                    for (size_t i = 0; i < n; ++i)
                    {
                        float x = (static_cast<float>(i) + 0.5f) / n;
                        float val = static_cast<float>(data(i));
                        float height = (val - y_range.x) / (y_range.y - y_range.x + 1e-6f) * m_data_rect.size.y;
                        if (height < 0) height = 0;
                        
                        godot::Vector2 top_left = m_data_rect.position + godot::Vector2(x * m_data_rect.size.x - bar_width/2, m_data_rect.size.y - height);
                        godot::Vector2 bottom_right = top_left + godot::Vector2(bar_width, height);
                        
                        draw_rect_canvas(godot::Rect2(top_left, bottom_right - top_left), m_line_color, true);
                        draw_rect_canvas(godot::Rect2(top_left, bottom_right - top_left), godot::Color(1,1,1), false);
                    }
                }
                
                void draw_histogram(const xarray_container<double>& data)
                {
                    size_t bins = std::min(static_cast<size_t>(50), static_cast<size_t>(std::sqrt(data.size())));
                    if (bins < 5) bins = 10;
                    
                    double min_val = xt::amin(data)();
                    double max_val = xt::amax(data)();
                    if (min_val == max_val) { min_val -= 0.5; max_val += 0.5; }
                    double bin_width = (max_val - min_val) / bins;
                    
                    std::vector<size_t> counts(bins, 0);
                    for (size_t i = 0; i < data.size(); ++i)
                    {
                        size_t bin = static_cast<size_t>((data.flat(i) - min_val) / bin_width);
                        if (bin >= bins) bin = bins - 1;
                        counts[bin]++;
                    }
                    
                    size_t max_count = *std::max_element(counts.begin(), counts.end());
                    float bar_width = m_data_rect.size.x / bins;
                    
                    for (size_t i = 0; i < bins; ++i)
                    {
                        float height = (static_cast<float>(counts[i]) / max_count) * m_data_rect.size.y;
                        godot::Vector2 top_left = m_data_rect.position + godot::Vector2(i * bar_width, m_data_rect.size.y - height);
                        draw_rect_canvas(godot::Rect2(top_left, godot::Vector2(bar_width-1, height)), m_line_color, true);
                    }
                }
                
                void draw_heatmap(const xarray_container<double>& data)
                {
                    if (data.dimension() != 2) return;
                    size_t h = data.shape()[0];
                    size_t w = data.shape()[1];
                    
                    double min_val = xt::amin(data)();
                    double max_val = xt::amax(data)();
                    if (min_val == max_val) { min_val -= 0.5; max_val += 0.5; }
                    
                    float cell_w = m_data_rect.size.x / w;
                    float cell_h = m_data_rect.size.y / h;
                    
                    for (size_t y = 0; y < h; ++y)
                    {
                        for (size_t x = 0; x < w; ++x)
                        {
                            double val = data(y, x);
                            double t = (val - min_val) / (max_val - min_val);
                            graphics::Color c = m_cmap->map(t, 0.0, 1.0);
                            godot::Color gd_color(c.r, c.g, c.b, c.a);
                            
                            godot::Vector2 top_left = m_data_rect.position + godot::Vector2(x * cell_w, y * cell_h);
                            draw_rect_canvas(godot::Rect2(top_left, godot::Vector2(cell_w, cell_h)), gd_color, true);
                        }
                    }
                }
                
                void compute_ranges(const xarray_container<double>& data, godot::Vector2& x_range, godot::Vector2& y_range)
                {
                    if (m_auto_y_range)
                    {
                        double ymin = xt::amin(data)();
                        double ymax = xt::amax(data)();
                        double padding = (ymax - ymin) * 0.05;
                        y_range = godot::Vector2(ymin - padding, ymax + padding);
                    }
                    else
                    {
                        y_range = m_y_range;
                    }
                    
                    if (m_auto_x_range)
                    {
                        x_range = godot::Vector2(0, static_cast<float>(data.shape()[0] - 1));
                    }
                    else
                    {
                        x_range = m_x_range;
                    }
                }
                
                // Canvas drawing helpers (pixel-based on Image)
                void draw_line_canvas(const godot::Vector2& from, const godot::Vector2& to, const godot::Color& color, float width)
                {
                    int x0 = from.x, y0 = from.y, x1 = to.x, y1 = to.y;
                    int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
                    int dy = -std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
                    int err = dx + dy;
                    
                    uint32_t col = color.to_rgba32();
                    int w = static_cast<int>(width/2);
                    
                    while (true)
                    {
                        for (int dyw = -w; dyw <= w; ++dyw)
                            for (int dxw = -w; dxw <= w; ++dxw)
                                m_canvas_image->set_pixel(x0 + dxw, y0 + dyw, col);
                        
                        if (x0 == x1 && y0 == y1) break;
                        int e2 = 2 * err;
                        if (e2 >= dy) { err += dy; x0 += sx; }
                        if (e2 <= dx) { err += dx; y0 += sy; }
                    }
                }
                
                void draw_circle_canvas(const godot::Vector2& center, float radius, const godot::Color& color)
                {
                    int cx = center.x, cy = center.y, r = radius;
                    uint32_t col = color.to_rgba32();
                    for (int y = -r; y <= r; ++y)
                        for (int x = -r; x <= r; ++x)
                            if (x*x + y*y <= r*r)
                                m_canvas_image->set_pixel(cx + x, cy + y, col);
                }
                
                void draw_rect_canvas(const godot::Rect2& rect, const godot::Color& color, bool filled)
                {
                    int x0 = rect.position.x, y0 = rect.position.y;
                    int x1 = x0 + rect.size.x, y1 = y0 + rect.size.y;
                    uint32_t col = color.to_rgba32();
                    for (int y = y0; y < y1; ++y)
                        for (int x = x0; x < x1; ++x)
                            m_canvas_image->set_pixel(x, y, col);
                }
                
                void draw_polygon_canvas(const std::vector<godot::Vector2>& points, const godot::Color& color)
                {
                    // Simple scanline fill for convex polygons
                    if (points.size() < 3) return;
                    int min_y = points[0].y, max_y = points[0].y;
                    for (const auto& p : points) { min_y = std::min(min_y, (int)p.y); max_y = std::max(max_y, (int)p.y); }
                    uint32_t col = color.to_rgba32();
                    for (int y = min_y; y <= max_y; ++y)
                    {
                        std::vector<int> intersections;
                        for (size_t i = 0; i < points.size(); ++i)
                        {
                            const auto& p1 = points[i];
                            const auto& p2 = points[(i+1)%points.size()];
                            if ((p1.y <= y && p2.y > y) || (p2.y <= y && p1.y > y))
                            {
                                float t = (y - p1.y) / (p2.y - p1.y);
                                int x = p1.x + t * (p2.x - p1.x);
                                intersections.push_back(x);
                            }
                        }
                        std::sort(intersections.begin(), intersections.end());
                        for (size_t i = 0; i < intersections.size(); i += 2)
                        {
                            if (i+1 < intersections.size())
                                for (int x = intersections[i]; x < intersections[i+1]; ++x)
                                    m_canvas_image->set_pixel(x, y, col);
                        }
                    }
                }
            };

            // --------------------------------------------------------------------
            // XTensorImage - Display tensor as image
            // --------------------------------------------------------------------
            class XTensorImage : public godot::TextureRect
            {
                GDCLASS(XTensorImage, godot::TextureRect)

            private:
                godot::Ref<XTensorNode> m_data_tensor;
                godot::String m_colormap = "viridis";
                bool m_auto_update = true;
                bool m_normalize = true;
                float m_exposure = 1.0f;
                float m_gamma = 2.2f;

            protected:
                static void _bind_methods()
                {
                    godot::ClassDB::bind_method(godot::D_METHOD("set_data_tensor", "tensor"), &XTensorImage::set_data_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_data_tensor"), &XTensorImage::get_data_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_colormap", "name"), &XTensorImage::set_colormap);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_colormap"), &XTensorImage::get_colormap);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_auto_update", "enabled"), &XTensorImage::set_auto_update);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_auto_update"), &XTensorImage::get_auto_update);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_normalize", "enabled"), &XTensorImage::set_normalize);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_normalize"), &XTensorImage::get_normalize);
                    godot::ClassDB::bind_method(godot::D_METHOD("refresh"), &XTensorImage::refresh);
                    
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::OBJECT, "data_tensor", godot::PROPERTY_HINT_RESOURCE_TYPE, "XTensorResource"), "set_data_tensor", "get_data_tensor");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::STRING, "colormap"), "set_colormap", "get_colormap");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::BOOL, "auto_update"), "set_auto_update", "get_auto_update");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::BOOL, "normalize"), "set_normalize", "get_normalize");
                }

            public:
                void set_data_tensor(const godot::Ref<XTensorNode>& tensor)
                {
                    m_data_tensor = tensor;
                    if (m_auto_update) refresh();
                }
                
                godot::Ref<XTensorNode> get_data_tensor() const { return m_data_tensor; }
                
                void set_colormap(const godot::String& name) { m_colormap = name; if (m_auto_update) refresh(); }
                godot::String get_colormap() const { return m_colormap; }
                
                void set_auto_update(bool enabled) { m_auto_update = enabled; }
                bool get_auto_update() const { return m_auto_update; }
                
                void set_normalize(bool enabled) { m_normalize = enabled; if (m_auto_update) refresh(); }
                bool get_normalize() const { return m_normalize; }
                
                void refresh()
                {
                    if (!m_data_tensor.is_valid() || !m_data_tensor->is_valid())
                    {
                        set_texture(godot::Ref<godot::Texture2D>());
                        return;
                    }
                    
                    auto data = m_data_tensor->get_tensor_resource()->m_data.to_double_array();
                    if (data.dimension() != 2 && data.dimension() != 3)
                    {
                        godot::UtilityFunctions::printerr("XTensorImage: data must be 2D (HxW) or 3D (HxWxC)");
                        return;
                    }
                    
                    size_t h = data.shape()[0];
                    size_t w = data.shape()[1];
                    size_t c = (data.dimension() == 3) ? data.shape()[2] : 1;
                    
                    godot::Ref<godot::Image> img;
                    img.instantiate();
                    
                    if (c == 1)
                    {
                        // Grayscale with colormap
                        img->create(w, h, false, godot::Image::FORMAT_RGB8);
                        double min_val = m_normalize ? xt::amin(data)() : 0.0;
                        double max_val = m_normalize ? xt::amax(data)() : 1.0;
                        if (max_val == min_val) max_val = min_val + 1.0;
                        
                        auto cmap = graphics::get_colormap(m_colormap.utf8().get_data());
                        for (size_t y = 0; y < h; ++y)
                        {
                            for (size_t x = 0; x < w; ++x)
                            {
                                double val = data(y, x);
                                double t = (val - min_val) / (max_val - min_val);
                                t = std::clamp(t, 0.0, 1.0);
                                graphics::Color col = cmap->map(t, 0.0, 1.0);
                                img->set_pixel(x, y, godot::Color(col.r, col.g, col.b));
                            }
                        }
                    }
                    else if (c == 3)
                    {
                        img->create(w, h, false, godot::Image::FORMAT_RGB8);
                        for (size_t y = 0; y < h; ++y)
                        {
                            for (size_t x = 0; x < w; ++x)
                            {
                                float r = std::clamp(static_cast<float>(data(y, x, 0)), 0.0f, 1.0f);
                                float g = std::clamp(static_cast<float>(data(y, x, 1)), 0.0f, 1.0f);
                                float b = std::clamp(static_cast<float>(data(y, x, 2)), 0.0f, 1.0f);
                                img->set_pixel(x, y, godot::Color(r, g, b));
                            }
                        }
                    }
                    else if (c == 4)
                    {
                        img->create(w, h, false, godot::Image::FORMAT_RGBA8);
                        for (size_t y = 0; y < h; ++y)
                        {
                            for (size_t x = 0; x < w; ++x)
                            {
                                float r = std::clamp(static_cast<float>(data(y, x, 0)), 0.0f, 1.0f);
                                float g = std::clamp(static_cast<float>(data(y, x, 1)), 0.0f, 1.0f);
                                float b = std::clamp(static_cast<float>(data(y, x, 2)), 0.0f, 1.0f);
                                float a = std::clamp(static_cast<float>(data(y, x, 3)), 0.0f, 1.0f);
                                img->set_pixel(x, y, godot::Color(r, g, b, a));
                            }
                        }
                    }
                    
                    godot::Ref<godot::ImageTexture> tex = godot::ImageTexture::create_from_image(img);
                    set_texture(tex);
                }
            };

            // --------------------------------------------------------------------
            // XGUIContainer - Layout container driven by tensor data
            // --------------------------------------------------------------------
            class XGUIContainer : public godot::Container
            {
                GDCLASS(XGUIContainer, godot::Container)

            private:
                godot::Ref<XTensorNode> m_layout_tensor; // N x 4: [x, y, w, h] normalized
                bool m_auto_update = true;

            protected:
                static void _bind_methods()
                {
                    godot::ClassDB::bind_method(godot::D_METHOD("set_layout_tensor", "tensor"), &XGUIContainer::set_layout_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_layout_tensor"), &XGUIContainer::get_layout_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_auto_update", "enabled"), &XGUIContainer::set_auto_update);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_auto_update"), &XGUIContainer::get_auto_update);
                    godot::ClassDB::bind_method(godot::D_METHOD("apply_layout"), &XGUIContainer::apply_layout);
                    
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::OBJECT, "layout_tensor", godot::PROPERTY_HINT_RESOURCE_TYPE, "XTensorResource"), "set_layout_tensor", "get_layout_tensor");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::BOOL, "auto_update"), "set_auto_update", "get_auto_update");
                }

            public:
                void _ready() override
                {
                    if (m_auto_update) apply_layout();
                }
                
                void _notification(int what)
                {
                    if (what == NOTIFICATION_SORT_CHILDREN)
                    {
                        apply_layout();
                    }
                }

                void set_layout_tensor(const godot::Ref<XTensorNode>& tensor)
                {
                    m_layout_tensor = tensor;
                    if (m_auto_update) queue_sort();
                }
                
                godot::Ref<XTensorNode> get_layout_tensor() const { return m_layout_tensor; }
                
                void set_auto_update(bool enabled) { m_auto_update = enabled; }
                bool get_auto_update() const { return m_auto_update; }
                
                void apply_layout()
                {
                    if (!m_layout_tensor.is_valid() || !m_layout_tensor->is_valid())
                    {
                        // Default layout: just let children keep their positions
                        return;
                    }
                    
                    auto layout = m_layout_tensor->get_tensor_resource()->m_data.to_double_array();
                    if (layout.dimension() != 2 || layout.shape()[1] < 4) return;
                    
                    godot::Vector2 container_size = get_size();
                    godot::TypedArray<godot::Node> children = get_children();
                    size_t n = std::min(static_cast<size_t>(children.size()), layout.shape()[0]);
                    
                    for (size_t i = 0; i < n; ++i)
                    {
                        godot::Control* ctrl = godot::Object::cast_to<godot::Control>(children[i]);
                        if (!ctrl) continue;
                        
                        float x = layout(i, 0) * container_size.x;
                        float y = layout(i, 1) * container_size.y;
                        float w = layout(i, 2) * container_size.x;
                        float h = layout(i, 3) * container_size.y;
                        
                        ctrl->set_position(godot::Vector2(x, y));
                        ctrl->set_size(godot::Vector2(w, h));
                    }
                }
            };

            // --------------------------------------------------------------------
            // XTensorText - Batch text rendering using tensors
            // --------------------------------------------------------------------
            class XTensorText : public godot::Control
            {
                GDCLASS(XTensorText, godot::Control)

            private:
                godot::Ref<XTensorNode> m_text_data; // N x (string length? Actually strings can't be in tensor directly, we use index to string table)
                godot::PackedStringArray m_string_table;
                godot::Ref<XTensorNode> m_positions; // N x 2
                godot::Ref<XTensorNode> m_colors;    // N x 4
                godot::Ref<XTensorNode> m_sizes;      // N
                godot::Ref<godot::Font> m_font;
                int m_font_size = 16;
                godot::Color m_default_color = godot::Color(1, 1, 1, 1);
                bool m_auto_update = true;

            protected:
                static void _bind_methods()
                {
                    godot::ClassDB::bind_method(godot::D_METHOD("set_string_table", "table"), &XTensorText::set_string_table);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_text_data", "tensor"), &XTensorText::set_text_data);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_positions", "tensor"), &XTensorText::set_positions);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_colors", "tensor"), &XTensorText::set_colors);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_sizes", "tensor"), &XTensorText::set_sizes);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_font", "font"), &XTensorText::set_font);
                    godot::ClassDB::bind_method(godot::D_METHOD("refresh"), &XTensorText::refresh);
                    
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::PACKED_STRING_ARRAY, "string_table"), "set_string_table", "get_string_table");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::OBJECT, "text_data", godot::PROPERTY_HINT_RESOURCE_TYPE, "XTensorResource"), "set_text_data", "get_text_data");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::OBJECT, "positions", godot::PROPERTY_HINT_RESOURCE_TYPE, "XTensorResource"), "set_positions", "get_positions");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::OBJECT, "colors", godot::PROPERTY_HINT_RESOURCE_TYPE, "XTensorResource"), "set_colors", "get_colors");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::OBJECT, "font", godot::PROPERTY_HINT_RESOURCE_TYPE, "Font"), "set_font", "get_font");
                }

            public:
                void _draw() override
                {
                    if (!m_text_data.is_valid() || !m_positions.is_valid()) return;
                    
                    auto indices = m_text_data->get_tensor_resource()->m_data.to_int_array();
                    auto pos = m_positions->get_tensor_resource()->m_data.to_double_array();
                    auto colors = m_colors.is_valid() ? m_colors->get_tensor_resource()->m_data.to_double_array() : xarray_container<double>();
                    auto sizes = m_sizes.is_valid() ? m_sizes->get_tensor_resource()->m_data.to_double_array() : xarray_container<double>();
                    
                    size_t n = std::min(indices.size(), pos.shape()[0]);
                    
                    for (size_t i = 0; i < n; ++i)
                    {
                        int idx = static_cast<int>(indices(i));
                        if (idx < 0 || idx >= m_string_table.size()) continue;
                        
                        godot::String text = m_string_table[idx];
                        godot::Vector2 p(pos(i,0), pos(i,1));
                        godot::Color col = m_default_color;
                        if (i < colors.shape()[0])
                            col = godot::Color(colors(i,0), colors(i,1), colors(i,2), colors.shape()[1] > 3 ? colors(i,3) : 1.0);
                        
                        int size = m_font_size;
                        if (i < sizes.size())
                            size = static_cast<int>(sizes(i));
                        
                        if (m_font.is_valid())
                            draw_string(m_font, p, text, godot::HORIZONTAL_ALIGNMENT_LEFT, -1, size, col);
                        else
                            draw_string(get_theme_default_font(), p, text, godot::HORIZONTAL_ALIGNMENT_LEFT, -1, size, col);
                    }
                }

                void set_string_table(const godot::PackedStringArray& table) { m_string_table = table; queue_redraw(); }
                godot::PackedStringArray get_string_table() const { return m_string_table; }
                
                void set_text_data(const godot::Ref<XTensorNode>& t) { m_text_data = t; queue_redraw(); }
                godot::Ref<XTensorNode> get_text_data() const { return m_text_data; }
                
                void set_positions(const godot::Ref<XTensorNode>& t) { m_positions = t; queue_redraw(); }
                godot::Ref<XTensorNode> get_positions() const { return m_positions; }
                
                void set_colors(const godot::Ref<XTensorNode>& t) { m_colors = t; queue_redraw(); }
                godot::Ref<XTensorNode> get_colors() const { return m_colors; }
                
                void set_sizes(const godot::Ref<XTensorNode>& t) { m_sizes = t; queue_redraw(); }
                godot::Ref<XTensorNode> get_sizes() const { return m_sizes; }
                
                void set_font(const godot::Ref<godot::Font>& font) { m_font = font; queue_redraw(); }
                godot::Ref<godot::Font> get_font() const { return m_font; }
                
                void refresh() { queue_redraw(); }
            };
#endif // XTENSOR_GODOT_CLASSDB_AVAILABLE

            // --------------------------------------------------------------------
            // Registration
            // --------------------------------------------------------------------
            class XGUIRegistry
            {
            public:
                static void register_classes()
                {
#if XTENSOR_GODOT_CLASSDB_AVAILABLE
                    godot::ClassDB::register_class<XTensorPlot>();
                    godot::ClassDB::register_class<XTensorImage>();
                    godot::ClassDB::register_class<XGUIContainer>();
                    godot::ClassDB::register_class<XTensorText>();
#endif
                }
            };

        } // namespace godot_bridge

        using godot_bridge::GUIPropertyBinding;
        using godot_bridge::XGUIBindingManager;
        using godot_bridge::XTensorPlot;
        using godot_bridge::XTensorImage;
        using godot_bridge::XGUIContainer;
        using godot_bridge::XTensorText;
        using godot_bridge::XGUIRegistry;

    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XGUI_HPP

// godot/xgui.hpp 