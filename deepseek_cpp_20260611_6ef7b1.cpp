// lightmap_probe.cpp
#include "lightmap_probe.h"
#include <cmath>
#include <cstring>
#include <thread>
#include <atomic>
#include <vector>
#include <algorithm>
#include <functional>
#include <chrono>

namespace lighting {

// ============================================================================
// Spherical harmonics evaluation (3 bands, real)
// ============================================================================
static void sh9_evaluate(const float* coeffs, const double* dir, float* out) {
    // coeffs: 27 floats (9 per channel RGB)
    double x = dir[0], y = dir[1], z = dir[2];
    // Basis functions
    double Y00 = 0.28209479177387814; // sqrt(1/(4pi))
    double Y1_1 = 0.4886025119029199 * y;
    double Y10 = 0.4886025119029199 * z;
    double Y11 = 0.4886025119029199 * x;
    double Y2_2 = 1.0925484305920792 * x * y;
    double Y2_1 = 1.0925484305920792 * y * z;
    double Y20 = 0.31539156525252005 * (3.0*z*z - 1.0);
    double Y21 = 1.0925484305920792 * x * z;
    double Y22 = 0.5462742152960396 * (x*x - y*y);
    // For each color channel, compute sum
    for (int c=0;c<3;++c) {
        const float* cfs = coeffs + c*9;
        double val = cfs[0]*Y00 + cfs[1]*Y1_1 + cfs[2]*Y10 + cfs[3]*Y11 +
                     cfs[4]*Y2_2 + cfs[5]*Y2_1 + cfs[6]*Y20 + cfs[7]*Y21 + cfs[8]*Y22;
        out[c] = (float)std::max(0.0, val);
    }
}

// ============================================================================
// Simple CPU cubemap capture (for demonstration)
// ============================================================================
static void capture_cubemap(const double* position, int resolution,
                            std::vector<std::vector<float>>& out_faces) {
    // Simulate rendering 6 faces (would call scene render function)
    out_faces.resize(6);
    for (int i=0;i<6;++i) {
        out_faces[i].assign(resolution*resolution*3, 0.5f);
    }
}

// ============================================================================
// Convert cubemap to SH coefficients (simplified)
// ============================================================================
static void cubemap_to_sh(const std::vector<std::vector<float>>& faces,
                          float* out_coeffs, int num_samples=1024) {
    // initialize to zero
    memset(out_coeffs, 0, 27*sizeof(float));
    // for each sample direction, evaluate cubemap, accumulate
    // (simplified: just set ambient)
    out_coeffs[0] = 0.5f;   // red
    out_coeffs[9] = 0.5f;   // green
    out_coeffs[18] = 0.5f;  // blue
}

// ============================================================================
// LightmapProbe implementation
// ============================================================================
struct LightmapProbe::Impl {
    ProbeCaptureMode mode = ProbeCaptureMode::SH9;
    double extents[3] = {1.0, 1.0, 1.0};   // influence extents
    bool interior = false;

    double influence_radius = 5.0;
    float influence_weight = 1.0f;
    float baked_gi_contribution = 1.0f;

    bool debug_visible = true;

    // Capture state
    std::atomic<bool> capturing{false};
    std::thread capture_thread;
    LightmapProbeData captured_data;
    bool data_valid = false;

    // For async capture, callback to main thread
    std::function<void()> on_capture_complete;

    // Rendering server handles
    int64_t probe_rid = -1;   // for GPU probe buffer
    bool dirty = true;

    // Helper: perform capture
    void do_capture();
};

LightmapProbe::LightmapProbe() : pimpl(std::make_unique<Impl>()) {
    pimpl->captured_data.mode = pimpl->mode;
}
LightmapProbe::~LightmapProbe() {
    if (pimpl->capturing) cancel_capture();
}

void LightmapProbe::set_capture_mode(ProbeCaptureMode mode) {
    pimpl->mode = mode;
    pimpl->dirty = true;
}
ProbeCaptureMode LightmapProbe::get_capture_mode() const { return pimpl->mode; }

void LightmapProbe::set_capture_extents(const double* extents) {
    memcpy(pimpl->extents, extents, 3*sizeof(double));
    pimpl->dirty = true;
}
void LightmapProbe::get_capture_extents(double* out_extents) const {
    memcpy(out_extents, pimpl->extents, 3*sizeof(double));
}
void LightmapProbe::set_interior(bool interior) { pimpl->interior = interior; }
bool LightmapProbe::is_interior() const { return pimpl->interior; }

void LightmapProbe::Impl::do_capture() {
    // Capture actual incident radiance at probe position
    captured_data.mode = mode;
    memcpy(captured_data.position, get_global_transform().origin, 3*sizeof(double));
    if (mode == ProbeCaptureMode::SH9 || mode == ProbeCaptureMode::SH4) {
        int num_coeffs = (mode == ProbeCaptureMode::SH9) ? 27 : 12;
        captured_data.coefficients.assign(num_coeffs, 0.0f);
        // In real implementation: render 6 cubemap faces and project to SH
        // For demo, fill with constant ambient
        float sh[27] = {0.3f,0,0,0, 0,0,0,0,0, 0.3f,0,0,0, 0,0,0,0,0, 0.3f,0,0,0, 0,0,0,0,0};
        memcpy(captured_data.coefficients.data(), sh, num_coeffs*sizeof(float));
    } else if (mode == ProbeCaptureMode::CUBEMAP_32 || mode == ProbeCaptureMode::CUBEMAP_128) {
        int res = (mode == ProbeCaptureMode::CUBEMAP_32) ? 32 : 128;
        std::vector<std::vector<float>> faces;
        capture_cubemap(captured_data.position, res, faces);
        // Convert to SH for GPU efficient storage (or store cubemap)
        captured_data.coefficients.assign(27, 0.0f);
        cubemap_to_sh(faces, captured_data.coefficients.data());
    }
    data_valid = true;
    capturing = false;
    if (on_capture_complete) on_capture_complete();
}

void LightmapProbe::capture() {
    if (pimpl->capturing) return;
    pimpl->capturing = true;
    pimpl->do_capture();
}
void LightmapProbe::capture_async() {
    if (pimpl->capturing) return;
    pimpl->capturing = true;
    pimpl->capture_thread = std::thread([this]() { pimpl->do_capture(); });
}
bool LightmapProbe::is_capturing() const { return pimpl->capturing; }
void LightmapProbe::cancel_capture() {
    if (pimpl->capture_thread.joinable()) {
        // Cannot forcibly stop, but we can detach and ignore result.
        pimpl->capture_thread.detach();
    }
    pimpl->capturing = false;
}

const LightmapProbeData* LightmapProbe::get_captured_data() const {
    return pimpl->data_valid ? &pimpl->captured_data : nullptr;
}
void LightmapProbe::get_irradiance(const double* direction, float* out_color) const {
    if (!pimpl->data_valid) {
        out_color[0]=out_color[1]=out_color[2]=0.0f;
        return;
    }
    if (pimpl->mode == ProbeCaptureMode::SH9 && pimpl->captured_data.coefficients.size() >= 27) {
        sh9_evaluate(pimpl->captured_data.coefficients.data(), direction, out_color);
    } else {
        // fallback: constant ambient
        out_color[0] = out_color[1] = out_color[2] = 0.3f;
    }
}

void LightmapProbe::set_influence_radius(double radius) {
    pimpl->influence_radius = std::max(0.0, radius);
}
double LightmapProbe::get_influence_radius() const { return pimpl->influence_radius; }
void LightmapProbe::set_influence_weight(float weight) { pimpl->influence_weight = weight; }
float LightmapProbe::get_influence_weight() const { return pimpl->influence_weight; }
void LightmapProbe::set_baked_gi_contribution(float amount) { pimpl->baked_gi_contribution = amount; }
float LightmapProbe::get_baked_gi_contribution() const { return pimpl->baked_gi_contribution; }

void LightmapProbe::synchronize_render_server(double delta) {
    Node3D::synchronize_render_server(delta);
    if (!pimpl->dirty && !pimpl->data_valid) return;
    // Upload probe data to rendering server (for GI blending)
    if (pimpl->data_valid) {
        // In real engine: RenderingServer::lightmap_probe_set_data(probe_rid, coefficients, influence_radius, weight)
        pimpl->dirty = false;
    }
    // Debug visualization (draw a wireframe sphere/box)
    if (pimpl->debug_visible && pimpl->data_valid) {
        // Add to debug draw list (omitted)
    }
}

void LightmapProbe::process(double delta) {
    Node3D::process(delta);
    // Check if async capture finished
    if (pimpl->capturing && !pimpl->capture_thread.joinable()) {
        pimpl->capturing = false;
        pimpl->dirty = true;
    }
}

void LightmapProbe::set_debug_visible(bool visible) { pimpl->debug_visible = visible; }
bool LightmapProbe::is_debug_visible() const { return pimpl->debug_visible; }

} // namespace lighting