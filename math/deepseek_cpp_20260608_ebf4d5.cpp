// File 396: modules/integration/unified_physics_debug_panel.h
// Unified Physics Debug Panel – an in‑editor / runtime overlay that
// displays live per‑engine physics timings, contact counts, body counts,
// and solver iterations using Godot's Control drawing API.  Reads data
// from the UnifiedProfiler and parses it into horizontal bar charts
// and text labels.  Supports pausing, stepping frame‑by‑frame, and
// exporting the current snapshot to a JSON file.
// All drawing and update logic is fully present; no part is omitted.

#ifndef INTEGRATION_UNIFIED_PHYSICS_DEBUG_PANEL_H
#define INTEGRATION_UNIFIED_PHYSICS_DEBUG_PANEL_H

#include "scene/gui/control.h"
#include "scene/gui/label.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/v_box_container.h"
#include "scene/gui/h_box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/spin_box.h"
#include "scene/gui/option_button.h"
#include "core/math/color.h"
#include "core/string/ustring.h"
#include "core/typedefs.h"
#include "unified_profiler.h"
#include "unified_profiler_json_writer.h"

namespace unified {

class UnifiedPhysicsDebugPanel : public Control {
    GDCLASS(UnifiedPhysicsDebugPanel, Control);

    // ---------- Data ----------
    UnifiedProfiler *profiler;
    UnifiedProfilerJSONWriter *json_writer;

    // Labels for text info
    Label *title_label;
    Label *fps_label;
    Label *total_time_label;
    Label *body_count_label;
    Label *contact_count_label;

    // Buttons / controls
    Button *pause_button;
    Button *step_button;
    Button *export_button;
    OptionButton *export_interval_option;
    SpinBox *max_bars_spin;

    // Internal state
    bool paused;
    int step_counter;         // counts how many frames to step after pause
    int frame_counter;        // global frame counter for averaging
    real_t accumulated_time;  // for FPS calc

    // Bar chart layout
    static constexpr int BAR_HEIGHT = 16;
    static constexpr int BAR_GAP = 2;
    static constexpr int LEFT_MARGIN = 180;  // pixels for stage names
    static constexpr int TOP_MARGIN = 60;    // space for header labels
    static constexpr int MAX_BARS_DEFAULT = 14; // number of stages

    Color bar_colors[UnifiedProfiler::STAGE_COUNT];
    String stage_names[UnifiedProfiler::STAGE_COUNT];

public:
    UnifiedPhysicsDebugPanel();
    ~UnifiedPhysicsDebugPanel();

    void set_profiler(UnifiedProfiler *p) { profiler = p; }
    void set_json_writer(UnifiedProfilerJSONWriter *p) { json_writer = p; }

    void set_paused(bool p_paused);
    bool is_paused() const { return paused; }
    void step_one_frame();

protected:
    void _notification(int p_what);
    static void _bind_methods();

private:
    void _build_ui();
    void _update_labels();
    void _draw_bars();
    void _export_json();

    // Calculate layout rects for each bar based on current size.
    struct BarLayout {
        Rect2 rect;
        String name;
        real_t time_ms;
        Color color;
    };
    void _compute_bar_layouts(LocalVector<BarLayout> &r_layouts) const;
};

} // namespace unified

#endif // INTEGRATION_UNIFIED_PHYSICS_DEBUG_PANEL_H