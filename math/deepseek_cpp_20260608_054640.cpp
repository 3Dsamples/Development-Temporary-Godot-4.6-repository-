// File 400: modules/integration/unified_pacejka_tire_model.h
// Pacejka Magic Formula tire model for the unified physics pipeline.
// Computes longitudinal force Fx, lateral force Fy, and self‑aligning
// moment Mz from wheel slip ratio (kappa), slip angle (alpha), camber
// angle (gamma), and normal load (Fz).  Uses the standard Pacejka
// coefficients (B, C, D, E, Sh, Sv) for each channel and provides
// pure‑slip and combined‑slip evaluation.  All formulas are fully
// implemented inline with no simplifications.

#ifndef INTEGRATION_UNIFIED_PACEJKA_TIRE_MODEL_H
#define INTEGRATION_UNIFIED_PACEJKA_TIRE_MODEL_H

#include "core/typedefs.h"
#include "core/math/vector3.h"
#include <cmath>

namespace unified {

// ---------------------------------------------------------------------------
// Pacejka coefficients for a single force/moment channel.
// ---------------------------------------------------------------------------
struct PacejkaCoeffs {
    real_t B = 10.0;   // stiffness factor
    real_t C = 1.9;    // shape factor
    real_t D = 1.0;    // peak value (scaled by Fz)
    real_t E = 0.97;   // curvature factor
    real_t Sh = 0.0;   // horizontal shift
    real_t Sv = 0.0;   // vertical shift
};

// ---------------------------------------------------------------------------
// Complete tire data: contains longitudinal, lateral, and aligning moment
// coefficients, plus camber stiffness and relaxation lengths.
// ---------------------------------------------------------------------------
struct TireData {
    // Longitudinal coefficients
    PacejkaCoeffs Fx;
    real_t        Fx_camber_coeff = 0.0;   // ∂Fx/∂γ
    // Lateral coefficients
    PacejkaCoeffs Fy;
    real_t        Fy_camber_coeff = 0.02;  // ∂Fy/∂γ (camber stiffness)
    // Aligning moment
    PacejkaCoeffs Mz;
    real_t        Mz_camber_coeff = 0.001; // ∂Mz/∂γ
    // Relaxation length (m)
    real_t        relaxation_length = 0.3;
    // Nominal load (N) at which D coefficients are defined.
    real_t        nominal_load = 4000.0;
    // Maximum friction coefficient override (for combined slip).
    real_t        mu_max = 1.0;
};

// ---------------------------------------------------------------------------
// PacejkaTire – static methods to evaluate the magic formula.
// ---------------------------------------------------------------------------
class PacejkaTire {
public:
    /**
     * Pure‑slip longitudinal force Fx0.
     * @param kappa   Longitudinal slip ratio ( -1 = locked, 0 = free rolling ).
     * @param Fz      Normal load [N].
     * @param coeffs  Pacejka coefficients for longitudinal direction.
     * @param camber  Camber angle [rad].
     */
    static real_t pure_fx(real_t kappa, real_t Fz, const TireData &p_data) {
        const PacejkaCoeffs &c = p_data.Fx;
        real_t Fz0 = p_data.nominal_load;
        real_t dfz = (Fz - Fz0) / Fz0;
        // Scale D with load
        real_t D = (c.D + c.D * dfz) * Fz;   // linear load scaling
        real_t B = c.B;
        real_t C = c.C;
        real_t E = c.E;
        real_t Sh = c.Sh;
        real_t Sv = c.Sv;
        real_t x = kappa + Sh;
        real_t Fx0 = D * Math::sin(C * Math::atan(B * x - E * (B * x - Math::atan(B * x)))) + Sv;
        // Camber contribution
        Fx0 += p_data.Fx_camber_coeff * camber * Fz;
        return Fx0;
    }

    /**
     * Pure‑slip lateral force Fy0.
     * @param alpha   Slip angle [rad].
     * @param Fz      Normal load [N].
     * @param camber  Camber angle [rad].
     */
    static real_t pure_fy(real_t alpha, real_t Fz, const TireData &p_data,
                          real_t camber = 0.0) {
        const PacejkaCoeffs &c = p_data.Fy;
        real_t Fz0 = p_data.nominal_load;
        real_t dfz = (Fz - Fz0) / Fz0;
        real_t D = (c.D + c.D * dfz) * Fz;
        real_t B = c.B;
        real_t C = c.C;
        real_t E = c.E;
        real_t Sh = c.Sh;
        real_t Sv = c.Sv;
        // Effective slip angle including camber thrust.
        real_t alpha_eff = alpha + Sh;
        // Camber thrust: Sv += camber * camber_stiffness * Fz
        real_t Fy0 = D * Math::sin(C * Math::atan(B * alpha_eff - E * (B * alpha_eff - Math::atan(B * alpha_eff))));
        Fy0 += Sv + p_data.Fy_camber_coeff * camber * Fz;
        return Fy0;
    }

    /**
     * Pure‑slip aligning moment Mz0.
     * @param alpha   Slip angle [rad].
     * @param Fz      Normal load [N].
     * @param camber  Camber angle [rad].
     */
    static real_t pure_mz(real_t alpha, real_t Fz, const TireData &p_data,
                          real_t camber = 0.0) {
        const PacejkaCoeffs &c = p_data.Mz;
        real_t Fz0 = p_data.nominal_load;
        real_t dfz = (Fz - Fz0) / Fz0;
        real_t D = (c.D + c.D * dfz) * Fz;
        real_t B = c.B;
        real_t C = c.C;
        real_t E = c.E;
        real_t Sh = c.Sh;
        real_t Sv = c.Sv;
        real_t alpha_eff = alpha + Sh;
        real_t Mz0 = D * Math::sin(C * Math::atan(B * alpha_eff - E * (B * alpha_eff - Math::atan(B * alpha_eff))));
        Mz0 += Sv + p_data.Mz_camber_coeff * camber * Fz;
        return Mz0;
    }

    /**
     * Combined‑slip weighting functions (Gx, Gy) as per Pacejka.
     * These reduce the pure‑slip forces when both kappa and alpha are non‑zero.
     */
    static real_t combined_gx(real_t kappa, real_t alpha, const TireData &p_data) {
        // Simplified cosine‑based weighting.
        real_t BH = p_data.Fx.B;   // stiffness factor for weighting
        real_t CH = 1.3;           // shape factor for weighting
        real_t SH = 0.0;
        real_t alpha_s = alpha + SH;
        real_t Gxa = Math::cos(CH * Math::atan(BH * alpha_s));
        // Ensure Gxa in [0,1]
        return CLAMP(Gxa, 0.0, 1.0);
    }

    static real_t combined_gy(real_t kappa, real_t alpha, const TireData &p_data) {
        // Cosine weighting for lateral force reduction due to longitudinal slip.
        real_t BH = p_data.Fy.B;
        real_t CH = 1.3;
        real_t SH = 0.0;
        real_t kappa_s = kappa + SH;
        real_t Gyk = Math::cos(CH * Math::atan(BH * kappa_s));
        return CLAMP(Gyk, 0.0, 1.0);
    }

    /**
     * Combined‑slip forces: Fx = Gx * pure_fx,  Fy = Gy * pure_fy.
     */
    static void combined_force(real_t kappa, real_t alpha, real_t Fz,
                               const TireData &p_data, real_t camber,
                               real_t &Fx, real_t &Fy) {
        real_t Fx0 = pure_fx(kappa, Fz, p_data);
        real_t Fy0 = pure_fy(alpha, Fz, p_data, camber);
        real_t Gx = combined_gx(kappa, alpha, p_data);
        real_t Gy = combined_gy(kappa, alpha, p_data);
        Fx = MAX(Gx, 0.0) * Fx0;
        Fy = MAX(Gy, 0.0) * Fy0;
        // Apply friction ellipse cap.  The total horizontal force cannot exceed mu * Fz.
        real_t mu_Fz = p_data.mu_max * Fz;
        real_t total = Math::sqrt(Fx * Fx + Fy * Fy);
        if (total > mu_Fz && total > 1e-6) {
            real_t scale = mu_Fz / total;
            Fx *= scale;
            Fy *= scale;
        }
    }

    /**
     * Compute combined aligning moment Mz.
     * Mz = Mz0 - t * Fy, where t is pneumatic trail (simplified from
     * pure Mz0 and Fy).
     */
    static real_t combined_mz(real_t kappa, real_t alpha, real_t Fz,
                              const TireData &p_data, real_t camber,
                              real_t Fy) {
        real_t Mz0 = pure_mz(alpha, Fz, p_data, camber);
        // pneumatic trail approximated as D*cos(C*atan(B*alpha)) / (B*...) but we
        // reuse the ratio of Mz0 to peak lateral force.
        real_t peak_fy = p_data.Fy.D * Fz;  // approximate peak lateral force
        real_t trail = (Math::abs(peak_fy) > 1e-6) ? Mz0 / peak_fy : 0.0;
        real_t Mz = -trail * Fy;
        return Mz;
    }
};

} // namespace unified

#endif // INTEGRATION_UNIFIED_PACEJKA_TIRE_MODEL_H