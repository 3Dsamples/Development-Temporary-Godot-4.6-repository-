// graphics/xgraphics.hpp

#ifndef XTENSOR_XGRAPHICS_HPP
#define XTENSOR_XGRAPHICS_HPP

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xexpression.hpp"
#include "../containers/xarray.hpp"
#include "../containers/xtensor.hpp"
#include "../core/xview.hpp"
#include "../core/xfunction.hpp"
#include "../io/xnpz.hpp"      // for saving plots as NPZ if needed
#include "../math/xstats.hpp"
#include "../math/xsorting.hpp"
#include "../math/xnorm.hpp"

#include <cmath>
#include <vector>
#include <string>
#include <map>
#include <functional>
#include <memory>
#include <algorithm>
#include <numeric>
#include <limits>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdint>
#include <tuple>
#include <array>
#include <utility>

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        namespace graphics
        {
            // --------------------------------------------------------------------
            // Color representation
            // --------------------------------------------------------------------
            struct Color
            {
                double r, g, b, a;
                
                Color(double red = 0.0, double green = 0.0, double blue = 0.0, double alpha = 1.0)
                    : r(red), g(green), b(blue), a(alpha) {}
                
                static Color from_hex(uint32_t hex, double alpha = 1.0)
                {
                    return Color(
                        ((hex >> 16) & 0xFF) / 255.0,
                        ((hex >> 8) & 0xFF) / 255.0,
                        (hex & 0xFF) / 255.0,
                        alpha
                    );
                }
                
                uint32_t to_rgba32() const
                {
                    uint32_t ri = static_cast<uint32_t>(std::clamp(r, 0.0, 1.0) * 255.0);
                    uint32_t gi = static_cast<uint32_t>(std::clamp(g, 0.0, 1.0) * 255.0);
                    uint32_t bi = static_cast<uint32_t>(std::clamp(b, 0.0, 1.0) * 255.0);
                    uint32_t ai = static_cast<uint32_t>(std::clamp(a, 0.0, 1.0) * 255.0);
                    return (ri << 24) | (gi << 16) | (bi << 8) | ai;
                }
            };
            
            // Predefined colors
            namespace colors
            {
                const Color red(1.0, 0.0, 0.0);
                const Color green(0.0, 1.0, 0.0);
                const Color blue(0.0, 0.0, 1.0);
                const Color cyan(0.0, 1.0, 1.0);
                const Color magenta(1.0, 0.0, 1.0);
                const Color yellow(1.0, 1.0, 0.0);
                const Color black(0.0, 0.0, 0.0);
                const Color white(1.0, 1.0, 1.0);
                const Color gray(0.5, 0.5, 0.5);
                const Color orange(1.0, 0.647, 0.0);
                const Color purple(0.5, 0.0, 0.5);
                const Color brown(0.647, 0.165, 0.165);
                const Color pink(1.0, 0.753, 0.796);
                const Color lime(0.0, 1.0, 0.0);
                const Color olive(0.5, 0.5, 0.0);
                const Color teal(0.0, 0.5, 0.5);
                const Color navy(0.0, 0.0, 0.5);
                const Color maroon(0.5, 0.0, 0.0);
                const Color silver(0.753, 0.753, 0.753);
            }
            
            // --------------------------------------------------------------------
            // Colormap interface
            // --------------------------------------------------------------------
            class Colormap
            {
            public:
                virtual ~Colormap() = default;
                virtual Color map(double value, double vmin = 0.0, double vmax = 1.0) const = 0;
                virtual std::string name() const = 0;
            };
            
            // Predefined colormaps
            class ViridisColormap : public Colormap
            {
            public:
                Color map(double value, double vmin, double vmax) const override
                {
                    double t = (value - vmin) / (vmax - vmin);
                    t = std::clamp(t, 0.0, 1.0);
                    // Simplified viridis approximation
                    double r = std::max(0.0, std::min(1.0, 0.267 * std::sin(t * 2.0 * 3.14159) + 0.5));
                    double g = std::max(0.0, std::min(1.0, 0.5 * std::sin(t * 2.0 * 3.14159 + 2.0) + 0.5));
                    double b = std::max(0.0, std::min(1.0, 0.7 * std::sin(t * 2.0 * 3.14159 + 4.0) + 0.5));
                    return Color(r, g, b);
                }
                std::string name() const override { return "viridis"; }
            };
            
            class PlasmaColormap : public Colormap
            {
            public:
                Color map(double value, double vmin, double vmax) const override
                {
                    double t = std::clamp((value - vmin) / (vmax - vmin), 0.0, 1.0);
                    double r = 0.5 + 0.5 * std::sin(t * 3.14159 * 2.0);
                    double g = 0.5 + 0.5 * std::cos(t * 3.14159 * 1.5);
                    double b = 0.5 + 0.5 * std::sin(t * 3.14159 * 2.5 + 1.0);
                    return Color(r, g, b);
                }
                std::string name() const override { return "plasma"; }
            };
            
            class JetColormap : public Colormap
            {
            public:
                Color map(double value, double vmin, double vmax) const override
                {
                    double t = std::clamp((value - vmin) / (vmax - vmin), 0.0, 1.0);
                    double r, g, b;
                    if (t < 0.25) {
                        r = 0.0; g = 0.0; b = 0.5 + 2.0 * t;
                    } else if (t < 0.5) {
                        r = 0.0; g = 4.0 * (t - 0.25); b = 1.0;
                    } else if (t < 0.75) {
                        r = 4.0 * (t - 0.5); g = 1.0; b = 1.0 - 4.0 * (t - 0.5);
                    } else {
                        r = 1.0; g = 1.0 - 2.0 * (t - 0.75); b = 0.0;
                    }
                    return Color(r, g, b);
                }
                std::string name() const override { return "jet"; }
            };
            
            class GrayColormap : public Colormap
            {
            public:
                Color map(double value, double vmin, double vmax) const override
                {
                    double t = std::clamp((value - vmin) / (vmax - vmin), 0.0, 1.0);
                    return Color(t, t, t);
                }
                std::string name() const override { return "gray"; }
            };
            
            // Factory for colormaps
            inline std::shared_ptr<Colormap> get_colormap(const std::string& name)
            {
                if (name == "viridis") return std::make_shared<ViridisColormap>();
                if (name == "plasma") return std::make_shared<PlasmaColormap>();
                if (name == "jet") return std::make_shared<JetColormap>();
                if (name == "gray" || name == "grey") return std::make_shared<GrayColormap>();
                return std::make_shared<ViridisColormap>();
            }
            
            // --------------------------------------------------------------------
            // Canvas abstraction (base class for rendering targets)
            // --------------------------------------------------------------------
            class Canvas
            {
            public:
                virtual ~Canvas() = default;
                virtual void clear(const Color& bg = colors::white) = 0;
                virtual void draw_line(double x0, double y0, double x1, double y1,
                                       const Color& color, double width = 1.0) = 0;
                virtual void draw_rect(double x, double y, double w, double h,
                                       const Color& color, bool filled = false) = 0;
                virtual void draw_circle(double cx, double cy, double r,
                                         const Color& color, bool filled = false) = 0;
                virtual void draw_text(double x, double y, const std::string& text,
                                       const Color& color, double size = 12.0) = 0;
                virtual void draw_polygon(const std::vector<std::pair<double, double>>& points,
                                          const Color& color, bool filled = false) = 0;
                virtual void draw_image(const xarray_container<uint8_t>& img,
                                        double x, double y, double w, double h) = 0;
                
                virtual std::pair<double, double> text_extents(const std::string& text, double size) const = 0;
                
                virtual std::pair<int, int> get_size() const = 0;
                virtual void save(const std::string& filename) = 0;
            };
            
            // A simple memory canvas that renders to an RGBA buffer
            class MemoryCanvas : public Canvas
            {
            public:
                MemoryCanvas(int width, int height, const Color& bg = colors::white)
                    : m_width(width), m_height(height), m_data(width * height * 4, 0)
                {
                    clear(bg);
                }
                
                void clear(const Color& bg) override
                {
                    uint8_t r = static_cast<uint8_t>(bg.r * 255);
                    uint8_t g = static_cast<uint8_t>(bg.g * 255);
                    uint8_t b = static_cast<uint8_t>(bg.b * 255);
                    uint8_t a = static_cast<uint8_t>(bg.a * 255);
                    for (size_t i = 0; i < m_data.size(); i += 4)
                    {
                        m_data[i] = r;
                        m_data[i+1] = g;
                        m_data[i+2] = b;
                        m_data[i+3] = a;
                    }
                }
                
                void draw_line(double x0, double y0, double x1, double y1,
                               const Color& color, double width) override
                {
                    // Bresenham's line algorithm with simple anti-aliasing (simplified)
                    int ix0 = static_cast<int>(x0);
                    int iy0 = static_cast<int>(y0);
                    int ix1 = static_cast<int>(x1);
                    int iy1 = static_cast<int>(y1);
                    
                    int dx = std::abs(ix1 - ix0);
                    int dy = -std::abs(iy1 - iy0);
                    int sx = ix0 < ix1 ? 1 : -1;
                    int sy = iy0 < iy1 ? 1 : -1;
                    int err = dx + dy;
                    
                    while (true)
                    {
                        if (ix0 >= 0 && ix0 < m_width && iy0 >= 0 && iy0 < m_height)
                        {
                            set_pixel(ix0, iy0, color);
                            if (width > 1.0)
                            {
                                int w = static_cast<int>(width/2);
                                for (int dxw = -w; dxw <= w; ++dxw)
                                    for (int dyw = -w; dyw <= w; ++dyw)
                                        if (ix0+dxw >= 0 && ix0+dxw < m_width && iy0+dyw >= 0 && iy0+dyw < m_height)
                                            set_pixel(ix0+dxw, iy0+dyw, color);
                            }
                        }
                        if (ix0 == ix1 && iy0 == iy1) break;
                        int e2 = 2 * err;
                        if (e2 >= dy) { err += dy; ix0 += sx; }
                        if (e2 <= dx) { err += dx; iy0 += sy; }
                    }
                }
                
                void draw_rect(double x, double y, double w, double h,
                               const Color& color, bool filled) override
                {
                    int ix = static_cast<int>(x);
                    int iy = static_cast<int>(y);
                    int iw = static_cast<int>(w);
                    int ih = static_cast<int>(h);
                    if (filled)
                    {
                        for (int dy = 0; dy < ih; ++dy)
                            for (int dx = 0; dx < iw; ++dx)
                                set_pixel(ix+dx, iy+dy, color);
                    }
                    else
                    {
                        draw_line(x, y, x+w, y, color);
                        draw_line(x+w, y, x+w, y+h, color);
                        draw_line(x+w, y+h, x, y+h, color);
                        draw_line(x, y+h, x, y, color);
                    }
                }
                
                void draw_circle(double cx, double cy, double r,
                                 const Color& color, bool filled) override
                {
                    int icx = static_cast<int>(cx);
                    int icy = static_cast<int>(cy);
                    int ir = static_cast<int>(r);
                    if (filled)
                    {
                        for (int dy = -ir; dy <= ir; ++dy)
                        {
                            for (int dx = -ir; dx <= ir; ++dx)
                            {
                                if (dx*dx + dy*dy <= ir*ir)
                                    set_pixel(icx+dx, icy+dy, color);
                            }
                        }
                    }
                    else
                    {
                        int x = 0, y = ir;
                        int d = 3 - 2 * ir;
                        while (y >= x)
                        {
                            set_pixel(icx+x, icy+y, color);
                            set_pixel(icx+y, icy+x, color);
                            set_pixel(icx-x, icy+y, color);
                            set_pixel(icx-y, icy+x, color);
                            set_pixel(icx+x, icy-y, color);
                            set_pixel(icx+y, icy-x, color);
                            set_pixel(icx-x, icy-y, color);
                            set_pixel(icx-y, icy-x, color);
                            x++;
                            if (d > 0) { y--; d = d + 4 * (x - y) + 10; }
                            else d = d + 4 * x + 6;
                        }
                    }
                }
                
                void draw_text(double x, double y, const std::string& text,
                               const Color& color, double size) override
                {
                    // Simplified: just draw a placeholder rectangle since font rendering is complex
                    // In a real implementation, use a font library like freetype
                    draw_rect(x, y - size, text.length() * size * 0.6, size, color, false);
                }
                
                void draw_polygon(const std::vector<std::pair<double, double>>& points,
                                  const Color& color, bool filled) override
                {
                    if (points.size() < 2) return;
                    if (filled && points.size() >= 3)
                    {
                        // Simple scanline fill for convex polygons
                        // Not implemented for brevity, just draw outline
                    }
                    for (size_t i = 0; i < points.size(); ++i)
                    {
                        size_t j = (i + 1) % points.size();
                        draw_line(points[i].first, points[i].second,
                                  points[j].first, points[j].second, color);
                    }
                }
                
                void draw_image(const xarray_container<uint8_t>& img,
                                double x, double y, double w, double h) override
                {
                    // img is expected to be HxWx3 or HxWx4
                    if (img.dimension() != 3) return;
                    size_t height = img.shape()[0];
                    size_t width = img.shape()[1];
                    size_t channels = img.shape()[2];
                    if (channels != 3 && channels != 4) return;
                    
                    for (size_t iy = 0; iy < height; ++iy)
                    {
                        for (size_t ix = 0; ix < width; ++ix)
                        {
                            double px = x + ix * w / width;
                            double py = y + iy * h / height;
                            int ipx = static_cast<int>(px);
                            int ipy = static_cast<int>(py);
                            if (ipx >= 0 && ipx < m_width && ipy >= 0 && ipy < m_height)
                            {
                                size_t idx = (iy * width + ix) * channels;
                                Color c;
                                c.r = img.data()[idx] / 255.0;
                                c.g = img.data()[idx+1] / 255.0;
                                c.b = img.data()[idx+2] / 255.0;
                                c.a = (channels == 4) ? img.data()[idx+3] / 255.0 : 1.0;
                                set_pixel(ipx, ipy, c);
                            }
                        }
                    }
                }
                
                std::pair<double, double> text_extents(const std::string& text, double size) const override
                {
                    return {text.length() * size * 0.6, size};
                }
                
                std::pair<int, int> get_size() const override
                {
                    return {m_width, m_height};
                }
                
                void save(const std::string& filename) override
                {
                    // Save as PPM or BMP for simplicity
                    std::ofstream file(filename, std::ios::binary);
                    if (!file) return;
                    // Simple PPM format
                    file << "P6\n" << m_width << " " << m_height << "\n255\n";
                    for (int y = 0; y < m_height; ++y)
                    {
                        for (int x = 0; x < m_width; ++x)
                        {
                            size_t idx = (y * m_width + x) * 4;
                            file.put(static_cast<char>(m_data[idx]));
                            file.put(static_cast<char>(m_data[idx+1]));
                            file.put(static_cast<char>(m_data[idx+2]));
                        }
                    }
                }
                
                const std::vector<uint8_t>& data() const { return m_data; }
                int width() const { return m_width; }
                int height() const { return m_height; }
                
            private:
                int m_width, m_height;
                std::vector<uint8_t> m_data;
                
                void set_pixel(int x, int y, const Color& color)
                {
                    if (x < 0 || x >= m_width || y < 0 || y >= m_height) return;
                    size_t idx = (y * m_width + x) * 4;
                    // Alpha blending
                    double alpha = color.a;
                    if (alpha < 1.0)
                    {
                        double bg_r = m_data[idx] / 255.0;
                        double bg_g = m_data[idx+1] / 255.0;
                        double bg_b = m_data[idx+2] / 255.0;
                        double r = color.r * alpha + bg_r * (1 - alpha);
                        double g = color.g * alpha + bg_g * (1 - alpha);
                        double b = color.b * alpha + bg_b * (1 - alpha);
                        m_data[idx] = static_cast<uint8_t>(r * 255);
                        m_data[idx+1] = static_cast<uint8_t>(g * 255);
                        m_data[idx+2] = static_cast<uint8_t>(b * 255);
                        m_data[idx+3] = 255;
                    }
                    else
                    {
                        m_data[idx] = static_cast<uint8_t>(color.r * 255);
                        m_data[idx+1] = static_cast<uint8_t>(color.g * 255);
                        m_data[idx+2] = static_cast<uint8_t>(color.b * 255);
                        m_data[idx+3] = static_cast<uint8_t>(color.a * 255);
                    }
                }
            };
            
            // --------------------------------------------------------------------
            // Axes class for managing coordinate transformations
            // --------------------------------------------------------------------
            class Axes
            {
            public:
                Axes(double x, double y, double width, double height)
                    : m_x(x), m_y(y), m_width(width), m_height(height)
                    , m_xlim(0.0, 1.0), m_ylim(0.0, 1.0)
                    , m_xscale("linear"), m_yscale("linear")
                    , m_title(""), m_xlabel(""), m_ylabel("")
                {
                }
                
                void set_xlim(double xmin, double xmax) { m_xlim = {xmin, xmax}; }
                void set_ylim(double ymin, double ymax) { m_ylim = {ymin, ymax}; }
                void set_xscale(const std::string& scale) { m_xscale = scale; }
                void set_yscale(const std::string& scale) { m_yscale = scale; }
                void set_title(const std::string& title) { m_title = title; }
                void set_xlabel(const std::string& label) { m_xlabel = label; }
                void set_ylabel(const std::string& label) { m_ylabel = label; }
                
                std::pair<double, double> transform(double x, double y) const
                {
                    double tx = transform_x(x);
                    double ty = transform_y(y);
                    return {m_x + m_width * tx, m_y + m_height * (1.0 - ty)};
                }
                
                double transform_x(double x) const
                {
                    if (m_xscale == "log")
                    {
                        if (x <= 0) x = m_xlim.first;
                        x = std::log10(x);
                        double log_min = std::log10(m_xlim.first);
                        double log_max = std::log10(m_xlim.second);
                        return (x - log_min) / (log_max - log_min);
                    }
                    return (x - m_xlim.first) / (m_xlim.second - m_xlim.first);
                }
                
                double transform_y(double y) const
                {
                    if (m_yscale == "log")
                    {
                        if (y <= 0) y = m_ylim.first;
                        y = std::log10(y);
                        double log_min = std::log10(m_ylim.first);
                        double log_max = std::log10(m_ylim.second);
                        return (y - log_min) / (log_max - log_min);
                    }
                    return (y - m_ylim.first) / (m_ylim.second - m_ylim.first);
                }
                
                void draw_frame(Canvas& canvas) const
                {
                    Color fg = colors::black;
                    // Draw bounding box
                    canvas.draw_rect(m_x, m_y, m_width, m_height, fg, false);
                    // Draw ticks and labels (simplified)
                }
                
                double get_x() const { return m_x; }
                double get_y() const { return m_y; }
                double get_width() const { return m_width; }
                double get_height() const { return m_height; }
                std::pair<double,double> get_xlim() const { return m_xlim; }
                std::pair<double,double> get_ylim() const { return m_ylim; }
                
            private:
                double m_x, m_y, m_width, m_height;
                std::pair<double,double> m_xlim, m_ylim;
                std::string m_xscale, m_yscale;
                std::string m_title, m_xlabel, m_ylabel;
            };
            
            // --------------------------------------------------------------------
            // Base Plot class
            // --------------------------------------------------------------------
            class Plot
            {
            public:
                virtual ~Plot() = default;
                virtual void draw(Canvas& canvas, const Axes& axes) const = 0;
                virtual std::string type() const = 0;
                virtual void set_color(const Color& c) { m_color = c; }
                virtual void set_label(const std::string& label) { m_label = label; }
                virtual void set_line_width(double w) { m_line_width = w; }
                virtual void set_marker(const std::string& m) { m_marker = m; }
                virtual void set_marker_size(double s) { m_marker_size = s; }
                
            protected:
                Color m_color = colors::blue;
                std::string m_label;
                double m_line_width = 1.5;
                std::string m_marker = "none";
                double m_marker_size = 6.0;
            };
            
            // Line plot
            template <class E1, class E2>
            class LinePlot : public Plot
            {
            public:
                LinePlot(const xexpression<E1>& x, const xexpression<E2>& y)
                    : m_x(eval(x.derived_cast())), m_y(eval(y.derived_cast()))
                {
                    if (m_x.size() != m_y.size())
                        XTENSOR_THROW(std::invalid_argument, "LinePlot: x and y must have same size");
                }
                
                void draw(Canvas& canvas, const Axes& axes) const override
                {
                    if (m_x.size() < 2) return;
                    auto prev = axes.transform(static_cast<double>(m_x(0)), static_cast<double>(m_y(0)));
                    for (size_t i = 1; i < m_x.size(); ++i)
                    {
                        auto curr = axes.transform(static_cast<double>(m_x(i)), static_cast<double>(m_y(i)));
                        canvas.draw_line(prev.first, prev.second, curr.first, curr.second,
                                         m_color, m_line_width);
                        prev = curr;
                    }
                    // Draw markers if any
                    if (m_marker != "none")
                    {
                        for (size_t i = 0; i < m_x.size(); ++i)
                        {
                            auto pt = axes.transform(static_cast<double>(m_x(i)), static_cast<double>(m_y(i)));
                            draw_marker(canvas, pt.first, pt.second);
                        }
                    }
                }
                
                std::string type() const override { return "line"; }
                
            private:
                xarray_container<double> m_x, m_y;
                
                void draw_marker(Canvas& canvas, double cx, double cy) const
                {
                    double r = m_marker_size / 2.0;
                    if (m_marker == "o")
                        canvas.draw_circle(cx, cy, r, m_color, false);
                    else if (m_marker == "s")
                        canvas.draw_rect(cx-r, cy-r, 2*r, 2*r, m_color, false);
                    else if (m_marker == "^")
                    {
                        std::vector<std::pair<double,double>> tri = {
                            {cx, cy-r}, {cx+r, cy+r}, {cx-r, cy+r}
                        };
                        canvas.draw_polygon(tri, m_color, false);
                    }
                    // else ignore
                }
            };
            
            // Scatter plot
            template <class E1, class E2>
            class ScatterPlot : public Plot
            {
            public:
                ScatterPlot(const xexpression<E1>& x, const xexpression<E2>& y)
                    : m_x(eval(x.derived_cast())), m_y(eval(y.derived_cast()))
                {
                    if (m_x.size() != m_y.size())
                        XTENSOR_THROW(std::invalid_argument, "ScatterPlot: x and y must have same size");
                }
                
                void draw(Canvas& canvas, const Axes& axes) const override
                {
                    for (size_t i = 0; i < m_x.size(); ++i)
                    {
                        auto pt = axes.transform(static_cast<double>(m_x(i)), static_cast<double>(m_y(i)));
                        canvas.draw_circle(pt.first, pt.second, m_marker_size/2.0, m_color, false);
                    }
                }
                
                std::string type() const override { return "scatter"; }
                
            private:
                xarray_container<double> m_x, m_y;
            };
            
            // Bar plot
            template <class E1, class E2>
            class BarPlot : public Plot
            {
            public:
                BarPlot(const xexpression<E1>& x, const xexpression<E2>& height, double width = 0.8)
                    : m_x(eval(x.derived_cast())), m_height(eval(height.derived_cast())), m_bar_width(width)
                {
                    if (m_x.size() != m_height.size())
                        XTENSOR_THROW(std::invalid_argument, "BarPlot: x and height must have same size");
                }
                
                void draw(Canvas& canvas, const Axes& axes) const override
                {
                    auto xlim = axes.get_xlim();
                    double x_span = xlim.second - xlim.first;
                    double bar_width_pixels = m_bar_width * x_span / m_x.size(); // approximate
                    
                    for (size_t i = 0; i < m_x.size(); ++i)
                    {
                        double x_val = static_cast<double>(m_x(i));
                        double h_val = static_cast<double>(m_height(i));
                        if (h_val == 0) continue;
                        auto bottom = axes.transform(x_val, 0.0);
                        auto top = axes.transform(x_val, h_val);
                        double x_center = bottom.first;
                        double bar_left = x_center - bar_width_pixels / 2.0 * axes.get_width();
                        double bar_right = x_center + bar_width_pixels / 2.0 * axes.get_width();
                        canvas.draw_rect(bar_left, top.second, bar_right - bar_left, bottom.second - top.second, m_color, true);
                    }
                }
                
                std::string type() const override { return "bar"; }
                
            private:
                xarray_container<double> m_x, m_height;
                double m_bar_width;
            };
            
            // Histogram plot (1D)
            template <class E>
            class HistPlot : public Plot
            {
            public:
                HistPlot(const xexpression<E>& data, size_t bins = 10)
                    : m_data(eval(data.derived_cast()))
                {
                    compute_histogram(bins);
                }
                
                void draw(Canvas& canvas, const Axes& axes) const override
                {
                    double bin_width = (m_bin_edges[1] - m_bin_edges[0]);
                    for (size_t i = 0; i < m_counts.size(); ++i)
                    {
                        double x_left = m_bin_edges[i];
                        double x_right = m_bin_edges[i+1];
                        double count = static_cast<double>(m_counts[i]);
                        auto bottom = axes.transform(x_left, 0.0);
                        auto top = axes.transform(x_left, count);
                        double bar_left = axes.transform_x(x_left) * axes.get_width() + axes.get_x();
                        double bar_right = axes.transform_x(x_right) * axes.get_width() + axes.get_x();
                        canvas.draw_rect(bar_left, top.second, bar_right - bar_left, bottom.second - top.second, m_color, true);
                    }
                }
                
                std::string type() const override { return "histogram"; }
                
            private:
                xarray_container<double> m_data;
                std::vector<double> m_bin_edges;
                std::vector<size_t> m_counts;
                
                void compute_histogram(size_t bins)
                {
                    double min_val = *std::min_element(m_data.begin(), m_data.end());
                    double max_val = *std::max_element(m_data.begin(), m_data.end());
                    double bin_width = (max_val - min_val) / bins;
                    m_bin_edges.resize(bins + 1);
                    for (size_t i = 0; i <= bins; ++i)
                        m_bin_edges[i] = min_val + i * bin_width;
                    m_counts.assign(bins, 0);
                    for (size_t i = 0; i < m_data.size(); ++i)
                    {
                        double val = m_data(i);
                        size_t idx = static_cast<size_t>((val - min_val) / bin_width);
                        if (idx >= bins) idx = bins - 1;
                        m_counts[idx]++;
                    }
                }
            };
            
            // Image plot (2D array as image)
            template <class E>
            class ImagePlot : public Plot
            {
            public:
                ImagePlot(const xexpression<E>& data, const std::string& cmap = "viridis")
                    : m_data(eval(data.derived_cast())), m_colormap(get_colormap(cmap))
                {
                    if (m_data.dimension() != 2)
                        XTENSOR_THROW(std::invalid_argument, "ImagePlot: data must be 2D");
                }
                
                void draw(Canvas& canvas, const Axes& axes) const override
                {
                    size_t h = m_data.shape()[0];
                    size_t w = m_data.shape()[1];
                    double vmin = *std::min_element(m_data.begin(), m_data.end());
                    double vmax = *std::max_element(m_data.begin(), m_data.end());
                    if (vmax == vmin) vmax = vmin + 1.0;
                    
                    double cell_w = axes.get_width() / w;
                    double cell_h = axes.get_height() / h;
                    double x0 = axes.get_x();
                    double y0 = axes.get_y();
                    
                    for (size_t i = 0; i < h; ++i)
                    {
                        for (size_t j = 0; j < w; ++j)
                        {
                            double val = static_cast<double>(m_data(i, j));
                            Color c = m_colormap->map(val, vmin, vmax);
                            canvas.draw_rect(x0 + j * cell_w, y0 + i * cell_h, cell_w, cell_h, c, true);
                        }
                    }
                }
                
                std::string type() const override { return "image"; }
                
            private:
                xarray_container<double> m_data;
                std::shared_ptr<Colormap> m_colormap;
            };
            
            // Contour plot (simplified)
            template <class E>
            class ContourPlot : public Plot
            {
            public:
                ContourPlot(const xexpression<E>& Z, size_t levels = 10)
                    : m_Z(eval(Z.derived_cast())), m_levels(levels)
                {
                    if (m_Z.dimension() != 2)
                        XTENSOR_THROW(std::invalid_argument, "ContourPlot: Z must be 2D");
                }
                
                void draw(Canvas& canvas, const Axes& axes) const override
                {
                    // Simplified: just draw a filled contour using image plot + lines for contour levels
                    // Not fully implemented, placeholder
                }
                
                std::string type() const override { return "contour"; }
                
            private:
                xarray_container<double> m_Z;
                size_t m_levels;
            };
            
            // --------------------------------------------------------------------
            // Figure class that contains multiple axes
            // --------------------------------------------------------------------
            class Figure
            {
            public:
                Figure(int width = 800, int height = 600)
                    : m_canvas(std::make_unique<MemoryCanvas>(width, height))
                {
                    // default single axes
                    add_axes(0.1, 0.1, 0.8, 0.8);
                }
                
                Axes& add_axes(double left, double bottom, double width, double height)
                {
                    double w = m_canvas->width();
                    double h = m_canvas->height();
                    m_axes.emplace_back(left * w, bottom * h, width * w, height * h);
                    return m_axes.back();
                }
                
                Axes& axes(size_t idx = 0) { return m_axes.at(idx); }
                const Axes& axes(size_t idx = 0) const { return m_axes.at(idx); }
                
                template <class E1, class E2>
                LinePlot<E1,E2>& plot(const xexpression<E1>& x, const xexpression<E2>& y, size_t ax_idx = 0)
                {
                    auto plot = std::make_unique<LinePlot<E1,E2>>(x, y);
                    auto& ref = *plot;
                    m_plots[ax_idx].push_back(std::move(plot));
                    return ref;
                }
                
                template <class E1, class E2>
                ScatterPlot<E1,E2>& scatter(const xexpression<E1>& x, const xexpression<E2>& y, size_t ax_idx = 0)
                {
                    auto plot = std::make_unique<ScatterPlot<E1,E2>>(x, y);
                    auto& ref = *plot;
                    m_plots[ax_idx].push_back(std::move(plot));
                    return ref;
                }
                
                template <class E1, class E2>
                BarPlot<E1,E2>& bar(const xexpression<E1>& x, const xexpression<E2>& height, size_t ax_idx = 0)
                {
                    auto plot = std::make_unique<BarPlot<E1,E2>>(x, height);
                    auto& ref = *plot;
                    m_plots[ax_idx].push_back(std::move(plot));
                    return ref;
                }
                
                template <class E>
                HistPlot<E>& hist(const xexpression<E>& data, size_t bins = 10, size_t ax_idx = 0)
                {
                    auto plot = std::make_unique<HistPlot<E>>(data, bins);
                    auto& ref = *plot;
                    m_plots[ax_idx].push_back(std::move(plot));
                    return ref;
                }
                
                template <class E>
                ImagePlot<E>& imshow(const xexpression<E>& data, const std::string& cmap = "viridis", size_t ax_idx = 0)
                {
                    auto plot = std::make_unique<ImagePlot<E>>(data, cmap);
                    auto& ref = *plot;
                    m_plots[ax_idx].push_back(std::move(plot));
                    return ref;
                }
                
                void clear()
                {
                    m_plots.clear();
                    m_axes.clear();
                    m_canvas->clear();
                }
                
                void draw()
                {
                    m_canvas->clear();
                    for (size_t i = 0; i < m_axes.size(); ++i)
                    {
                        m_axes[i].draw_frame(*m_canvas);
                        auto it = m_plots.find(i);
                        if (it != m_plots.end())
                        {
                            for (auto& plot : it->second)
                                plot->draw(*m_canvas, m_axes[i]);
                        }
                    }
                }
                
                void save(const std::string& filename)
                {
                    draw();
                    m_canvas->save(filename);
                }
                
                void show() // console placeholder
                {
                    draw();
                    // In real implementation, display in window or output to terminal
                    std::cout << "Figure (" << m_canvas->width() << "x" << m_canvas->height() << ") rendered." << std::endl;
                }
                
                Canvas& canvas() { return *m_canvas; }
                
            private:
                std::unique_ptr<MemoryCanvas> m_canvas;
                std::vector<Axes> m_axes;
                std::map<size_t, std::vector<std::unique_ptr<Plot>>> m_plots;
            };
            
            // --------------------------------------------------------------------
            // Convenience functions for quick plotting
            // --------------------------------------------------------------------
            template <class E1, class E2>
            inline void plot(const xexpression<E1>& x, const xexpression<E2>& y, const std::string& filename = "")
            {
                Figure fig;
                auto& line = fig.plot(x, y);
                line.set_color(colors::blue);
                if (!filename.empty())
                    fig.save(filename);
                else
                    fig.show();
            }
            
            template <class E>
            inline void imshow(const xexpression<E>& data, const std::string& cmap = "viridis")
            {
                Figure fig;
                fig.imshow(data, cmap);
                fig.show();
            }
            
            // --------------------------------------------------------------------
            // 3D Plotting (basic wireframe/surface)
            // --------------------------------------------------------------------
            template <class E1, class E2, class E3>
            class SurfacePlot : public Plot
            {
            public:
                SurfacePlot(const xexpression<E1>& X, const xexpression<E2>& Y, const xexpression<E3>& Z)
                    : m_X(eval(X.derived_cast())), m_Y(eval(Y.derived_cast())), m_Z(eval(Z.derived_cast()))
                {
                    if (m_X.dimension() != 2 || m_Y.dimension() != 2 || m_Z.dimension() != 2)
                        XTENSOR_THROW(std::invalid_argument, "SurfacePlot: X,Y,Z must be 2D grids");
                }
                
                void draw(Canvas& canvas, const Axes& axes) const override
                {
                    // Project 3D points to 2D using simple orthographic projection
                    size_t rows = m_X.shape()[0];
                    size_t cols = m_X.shape()[1];
                    for (size_t i = 0; i < rows-1; ++i)
                    {
                        for (size_t j = 0; j < cols-1; ++j)
                        {
                            // Draw quadrilaterals as two triangles
                            double x0 = static_cast<double>(m_X(i,j));
                            double y0 = static_cast<double>(m_Y(i,j));
                            double z0 = static_cast<double>(m_Z(i,j));
                            double x1 = static_cast<double>(m_X(i+1,j));
                            double y1 = static_cast<double>(m_Y(i+1,j));
                            double z1 = static_cast<double>(m_Z(i+1,j));
                            double x2 = static_cast<double>(m_X(i,j+1));
                            double y2 = static_cast<double>(m_Y(i,j+1));
                            double z2 = static_cast<double>(m_Z(i,j+1));
                            double x3 = static_cast<double>(m_X(i+1,j+1));
                            double y3 = static_cast<double>(m_Y(i+1,j+1));
                            double z3 = static_cast<double>(m_Z(i+1,j+1));
                            
                            auto p0 = project(axes, x0, y0, z0);
                            auto p1 = project(axes, x1, y1, z1);
                            auto p2 = project(axes, x2, y2, z2);
                            auto p3 = project(axes, x3, y3, z3);
                            
                            canvas.draw_line(p0.first, p0.second, p1.first, p1.second, m_color);
                            canvas.draw_line(p1.first, p1.second, p3.first, p3.second, m_color);
                            canvas.draw_line(p3.first, p3.second, p2.first, p2.second, m_color);
                            canvas.draw_line(p2.first, p2.second, p0.first, p0.second, m_color);
                        }
                    }
                }
                
                std::string type() const override { return "surface"; }
                
            private:
                xarray_container<double> m_X, m_Y, m_Z;
                
                std::pair<double,double> project(const Axes& axes, double x, double y, double z) const
                {
                    // Simple orthographic projection ignoring z (or use perspective)
                    return axes.transform(x, y);
                }
            };
            
            // --------------------------------------------------------------------
            // Utilities for auto-scaling and layout
            // --------------------------------------------------------------------
            inline void tight_layout(Figure& fig)
            {
                // Adjust subplot parameters to fit labels, not implemented
            }
            
            inline void subplots(size_t nrows, size_t ncols, std::vector<Figure>& figs)
            {
                // Create a grid of figures (each separate) - not implemented
            }
            
        } // namespace graphics
        
        // Bring graphics into xt namespace
        using graphics::Color;
        using graphics::colors;
        using graphics::Colormap;
        using graphics::Canvas;
        using graphics::MemoryCanvas;
        using graphics::Axes;
        using graphics::Plot;
        using graphics::LinePlot;
        using graphics::ScatterPlot;
        using graphics::BarPlot;
        using graphics::HistPlot;
        using graphics::ImagePlot;
        using graphics::ContourPlot;
        using graphics::SurfacePlot;
        using graphics::Figure;
        using graphics::plot;
        using graphics::imshow;
        
    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XGRAPHICS_HPP

// graphics/xgraphics.hpp