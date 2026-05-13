perceptual_color.h
/**
 * ============================================================================
 * PERCEPTUAL COLOR SYSTEM (STANDALONE)
 * ============================================================================
 * 
 * CORE FEATURES:
 * 1. Oklab/OKLCH Perceptual Spaces: Separates Lightness from Color.
 * 2. Perfect Tinting: Replaces Hue while preserving perceived structural detail.
 * 3. Pure Chromatization: Boosts vibrancy without "burning" pixels.
 * 4. High Performance: Optimized matrix transforms and cube-root math.
 * 5. Gamut Safety: Automatic clamping to ensure display-safe Linear RGB output.
 */

#ifndef PERCEPTUAL_COLOR_SYSTEM_H
#define PERCEPTUAL_COLOR_SYSTEM_H

#include <cmath>
#include <algorithm>
#include <iostream>

// ----------------------------------------------------------------------------
// DATA STRUCTURES
// ----------------------------------------------------------------------------

/**
 * @brief Standard 4-component vector for RGBA data (Linear Space).
 */
struct Vector4 {
    float r, g, b, a;

    Vector4() : r(0.0f), g(0.0f), b(0.0f), a(1.0f) {}
    Vector4(float _r, float _g, float _b, float _a) : r(_r), g(_g), b(_b), a(_a) {}
};

/**
 * @brief Oklab Cartesian coordinates.
 * L: Perceived Lightness (0.0 - 1.0)
 * a: Green-Red axis
 * b: Blue-Yellow axis
 */
struct ColorOklab {
    float L, a, b;
};

/**
 * @brief OKLCH Cylindrical (Polar) coordinates.
 * L: Perceived Lightness
 * C: Chroma (Vibrancy/Intensity)
 * H: Hue Angle in Radians (-PI to PI)
 */
struct ColorOKLCH {
    float L, C, H;
};

// ----------------------------------------------------------------------------
// PERCEPTUAL COLOR ENGINE CLASS
// ----------------------------------------------------------------------------

class PerceptualColorEngine {
public:
    // --- Core Conversions ---
    static ColorOklab  LinearRGBToOklab(Vector4 rgb);
    static Vector4     OklabToLinearRGB(ColorOklab lab, float alpha);
    static ColorOKLCH  OklabToOKLCH(ColorOklab lab);
    static ColorOklab  OKLCHToOklab(ColorOKLCH lch);

    // --- High-Level Operations ---
    
    /**
     * @brief The "Perfect Tint" Logic.
     * Replaces texture hue with target hue while preserving original lightness.
     * 
     * @param baseTexture The sampled color from the original texture.
     * @param tintColor The arbitrary color parameter for the tint.
     * @param factor Intensity of the effect (0.0 to 1.0).
     */
    static Vector4 ApplyPerfectTint(Vector4 baseTexture, Vector4 tintColor, float factor);

    /**
     * @brief Perceptual Chromatization.
     * Adjusts the 'purity' or 'vibrancy' of a color without shifting hue.
     * 
     * @param color Input Linear RGB color.
     * @param scale 1.0 = neutral, >1.0 = more vibrant, <1.0 = desaturated.
     */
    static Vector4 ApplyChromatization(Vector4 color, float scale);

private:
    static inline float Clamp01(float v) {
        return std::max(0.0f, std::min(1.0f, v));
    }
};

#endif // PERCEPTUAL_COLOR_SYSTEM_H

// ----------------------------------------------------------------------------
// IMPLEMENTATION SECTION
// ----------------------------------------------------------------------------

/**
 * PHASE 1 & 2: RGB to LMS to Oklab
 * We project RGB into a cone-response space (LMS), apply non-linear 
 * compression (cube root), and then project into Oklab coordinates.
 */
ColorOklab PerceptualColorEngine::LinearRGBToOklab(Vector4 rgb) {
    // LMS Matrix Transform (Björn Ottosson D65)
    float l = 0.4122214708f * rgb.r + 0.5363325363f * rgb.g + 0.0514459929f * rgb.b;
    float m = 0.2119034982f * rgb.r + 0.6806995451f * rgb.g + 0.1073969566f * rgb.b;
    float s = 0.0883024619f * rgb.r + 0.2817188376f * rgb.g + 0.6299787005f * rgb.b;

    // Perceptual compression using cube root
    float l_prime = std::cbrtf(l);
    float m_prime = std::cbrtf(m);
    float s_prime = std::cbrtf(s);

    ColorOklab res;
    res.L = 0.2104542553f * l_prime + 0.7936177850f * m_prime - 0.0040720468f * s_prime;
    res.a = 1.9779984951f * l_prime - 2.4285922050f * m_prime + 0.4505937099f * s_prime;
    res.b = 0.0259040371f * l_prime + 0.7827717662f * m_prime - 0.8086758033f * s_prime;
    return res;
}

/**
 * PHASE 5: Oklab to RGB with Gamut Clamping
 * Reverses the non-linear LMS compression and projects back to displayable RGB.
 */
Vector4 PerceptualColorEngine::OklabToLinearRGB(ColorOklab lab, float alpha) {
    float l_prime = lab.L + 0.3963377774f * lab.a + 0.2158037573f * lab.b;
    float m_prime = lab.L - 0.1055613458f * lab.a - 0.0638541728f * lab.b;
    float s_prime = lab.L - 0.0894841775f * lab.a - 1.2914855480f * lab.b;

    float l = l_prime * l_prime * l_prime;
    float m = m_prime * m_prime * m_prime;
    float s = s_prime * s_prime * s_prime;

    Vector4 res;
    res.r = +4.0767416621f * l - 3.3077115913f * m + 0.2309699292f * s;
    res.g = -1.2684380046f * l + 2.6097574011f * m - 0.3413193965f * s;
    res.b = -0.0041960863f * l - 0.7034186147f * m + 1.7076147010f * s;
    res.a = alpha;

    // Force values into [0, 1] gamut to avoid visual artifacts
    res.r = Clamp01(res.r);
    res.g = Clamp01(res.g);
    res.b = Clamp01(res.b);
    return res;
}

/**
 * PHASE 3: Cartesian to Polar
 */
ColorOKLCH PerceptualColorEngine::OklabToOKLCH(ColorOklab lab) {
    ColorOKLCH res;
    res.L = lab.L;
    res.C = std::sqrt(lab.a * lab.a + lab.b * lab.b);
    res.H = std::atan2(lab.b, lab.a);
    return res;
}

/**
 * Polar to Cartesian
 */
ColorOklab PerceptualColorEngine::OKLCHToOklab(ColorOKLCH lch) {
    ColorOklab res;
    res.L = lch.L;
    res.a = lch.C * std::cos(lch.H);
    res.b = lch.C * std::sin(lch.H);
    return res;
}

/**
 * PHASE 4: The Advanced Tint Logic
 * This is the core implementation of the vibrant tinting method.
 */
Vector4 PerceptualColorEngine::ApplyPerfectTint(Vector4 baseTexture, Vector4 tintColor, float factor) {
    // 1. Project both colors into the perceptual polar space
    ColorOKLCH baseLCH = OklabToOKLCH(LinearRGBToOklab(baseTexture));
    ColorOKLCH tintLCH = OklabToOKLCH(LinearRGBToOklab(tintColor));

    ColorOKLCH targetLCH;

    // 2. Structural Preservation: Carry over original texture lightness.
    targetLCH.L = baseLCH.L;

    // 3. Hue Replacement: Adopt the tint parameter's hue angle.
    targetLCH.H = tintLCH.H;

    // 4. Chroma Heuristic: 
    // We scale the texture's natural chroma by the "Saturation Ratio" of the tint.
    // This allows a slightly desaturated tint to yield a slightly desaturated texture.
    float tintSaturation = tintLCH.C / (tintLCH.L + 0.00001f);
    targetLCH.C = baseLCH.C * Clamp01(tintSaturation);

    // 5. Final Synthesis
    Vector4 tintedRGB = OklabToLinearRGB(OKLCHToOklab(targetLCH), baseTexture.a);

    // Alpha-aware Lerp
    Vector4 output;
    output.r = baseTexture.r + factor * (tintedRGB.r - baseTexture.r);
    output.g = baseTexture.g + factor * (tintedRGB.g - baseTexture.g);
    output.b = baseTexture.b + factor * (tintedRGB.b - baseTexture.b);
    output.a = baseTexture.a;

    return output;
}

/**
 * Robust Chromatization (Vibrancy Control)
 */
Vector4 PerceptualColorEngine::ApplyChromatization(Vector4 color, float scale) {
    ColorOklab lab = LinearRGBToOklab(color);
    ColorOKLCH lch = OklabToOKLCH(lab);

    // Strictly adjust Chroma while keeping Lightness and Hue constant.
    lch.C *= scale;

    return OklabToLinearRGB(OKLCHToOklab(lch), color.a);
}

// ----------------------------------------------------------------------------
// STANDALONE EXAMPLE / TEST
// ----------------------------------------------------------------------------

int main() {
    // 1. Setup a photorealistic "Forest Floor" pixel (Dark Green)
    Vector4 forestPixel(0.12f, 0.25f, 0.08f, 1.0f);

    // 2. Setup a vivid "Anime Sky" tint (Bright Cyan)
    Vector4 animeCyan(0.0f, 1.0f, 1.0f, 1.0f);

    // 3. Execute Perfect Tinting
    Vector4 finalColor = PerceptualColorEngine::ApplyPerfectTint(forestPixel, animeCyan, 1.0f);

    std::cout << "--- Perceptual Color System Execution ---" << std::endl;
    std::cout << "Base Pixel (Linear RGB): 0.12, 0.25, 0.08" << std::endl;
    std::cout << "Target Tint (Cyan):      0.00, 1.00, 1.00" << std::endl;
    std::cout << "Result (Perfect Tint):   " << finalColor.r << ", " << finalColor.g << ", " << finalColor.b << std::endl;
    std::cout << "Result (Vibrancy Boost): " << std::endl;

    // 4. Execute Chromatization
    Vector4 vibrant = PerceptualColorEngine::ApplyChromatization(forestPixel, 2.0f);
    std::cout << "Boosted (2.0x Chroma):   " << vibrant.r << ", " << vibrant.g << ", " << vibrant.b << std::endl;

    return 0;
}
