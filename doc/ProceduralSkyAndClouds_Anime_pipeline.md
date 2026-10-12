

## BATCH 14: ARCHITECTURAL EXPANSION OF PROCEDURALSKYANDCLOUDS.JS SUB-MODULES

To achieve a breathtaking, dynamic sky that blends realistic atmospheric physics with high-performance, stylized anime aesthetics, the `ProceduralSkyAndClouds.js` module is decomposed into 10 highly specialized sub-files. These files manage sky irradiance, biome-driven cloud formations, dynamic shadow casting, and sun occlusion, all optimized to maintain a locked 60+ FPS without relying on heavy, traditional volumetric raymarching.

<span style="color:red">SkyIrradianceAndScattering.js</span>
Imports: `ShaderMaterial.js`, `Vector3.js`, `Color.js`, `MathUtils.js`
Functions & Algorithms: `approximateRayleighScattering()`, `calculateMieScatteringGlow()`, `generateSkyGradient()`, `optimizeScatteringMath()`.
Context-Aware Integration: Computes the base sky color by approximating physical light scattering using optimized trigonometric functions. Instead of expensive exponential integrals, it uses a pre-calculated 1D lookup texture or a simplified polynomial approximation. The scattering intensity is context-aware: it increases the Mie scattering (the bright halo around the sun) during the Golden Hour preset and dampens it during the Stormy Tempest preset to create a flat, oppressive grey sky.

<span style="color:red">BiomeDrivenCloudFormation.js</span>
Imports: `MathUtils.js`, `Vector3.js`, `NoiseGenerator.js`, `Color.js`
Functions & Algorithms: `evaluateWorleyNoise()`, `calculateSimplexFractalBrownianMotion()`, `mapTemperatureToCloudType()`, `generateCloudDensityField()`.
Context-Aware Integration: Generates the foundational cloud shapes using a combination of Worley noise (for distinct, puffy cell boundaries) and Simplex fBm (for organic detail). The noise parameters are dynamically driven by the biome's simulated temperature and humidity. For the Arid Desert, it generates sparse, high-altitude cirrus clouds; for the Tropical Rainforest, it generates dense, low-altitude cumulonimbus formations with heavy vertical development.

<span style="color:red">CelShadedCloudMaterial.js</span>
Imports: `ShaderMaterial.js`, `Color.js`, `MathUtils.js`, `Vector3.js`
Functions & Algorithms: `applyStepFunctionShading()`, `calculateCloudRimLighting()`, `quantizeCloudAlbedo()`, `injectAnimeColorBands()`.
Context-Aware Integration: The core of the anime cloud aesthetic. Instead of smooth, realistic phase functions, it passes the calculated cloud density through a strict `step` or narrow `smoothstep` function to create hard, distinct color bands (e.g., pure white for highlights, soft grey for midtones, deep blue/purple for shadows). It also applies a stylized rim light to the top edges of the clouds, separating them sharply from the sky background.

<span style="color:red">CloudShadowProjector.js</span>
Imports: `WebGLRenderTarget.js`, `Matrix4.js`, `ShaderMaterial.js`, `Vector3.js`
Functions & Algorithms: `projectCloudDensityToGround()`, `calculateUmbraAndPenumbra()`, `applyScreenSpaceCloudShadows()`, `modulateShadowOpacity()`.
Context-Aware Integration: Crucial for environmental realism. Instead of casting shadows from millions of cloud particles, it renders the cloud density field to a low-resolution (e.g., 512x512) shadow map from the sun's perspective. This map is then projected onto the terrain in the main render pass. The shadow edges are quantized to match the anime style, and the shadow color is tinted with the ambient sky color to prevent muddy, black shadows on the ground.

<span style="color:red">SunOcclusionAndRayCaster.js</span>
Imports: `Vector3.js`, `Color.js`, `ShaderMaterial.js`, `MathUtils.js`
Functions & Algorithms: `calculateSunDiscIntersection()`, `evaluateCloudOcclusionFactor()`, `generateCrepuscularRays()`, `applyRadialBlurApproximation()`.
Context-Aware Integration: Dynamically blocks the sun disc when a dense cloud passes in front of it, instantly dimming the global directional light intensity. Simultaneously, it generates stylized "god rays" (crepuscular rays) by applying a screen-space radial blur originating from the sun's position, masked by the cloud shadow map. The ray intensity is context-aware, blazing brightly during the Morning Mist and completely disabled during the Overcast Storm.

<span style="color:red">DynamicWeatherCloudMorpher.js</span>
Imports: `MathUtils.js`, `Vector3.js`, `EventDispatcher.js`, `Clock.js`
Functions & Algorithms: `interpolateNoiseScales()`, `morphCloudDensity()`, `transitionCloudTypes()`, `smoothStepWeatherChanges()`.
Context-Aware Integration: Manages the seamless transformation of cloud shapes during climatic changes. When transitioning from Clear to Stormy, it mathematically interpolates the Worley noise scale and the fBm turbulence. The fluffy, distinct cumulus clouds slowly stretch, merge, and darken into a solid, rolling stratus layer. This morphing happens entirely in the shader via uniform interpolation, requiring zero CPU geometry updates.

<span style="color:red">TimeOfDaySkyPalette.js</span>
Imports: `Color.js`, `Vector3.js`, `MathUtils.js`, `Clock.js`
Functions & Algorithms: `calculateSolarElevationAngle()`, `interpolateSkyColorGradients()`, `shiftCloudTintByTime()`, `manageNightSkyTransition()`.
Context-Aware Integration: Drives the massive color shifts of the sky and clouds based on the time of day. It uses a 1D gradient texture or a series of cubic Bézier curves to interpolate the zenith, horizon, and sun colors. During sunset, it shifts the cloud undersides to a vibrant magenta and orange. At night, it transitions the sky to a deep indigo and generates a procedural starfield using a high-frequency hash function, while the clouds take on a dark, moonlit silver tint.

<span style="color:red">CloudWindAndAdvectionSystem.js</span>
Imports: `Vector3.js`, `MathUtils.js`, `Clock.js`, `ShaderMaterial.js`
Functions & Algorithms: `calculateAltitudeWindShear()`, `advectNoiseCoordinates()`, `animateCloudMorphing()`, `applyTurbulenceVectors()`.
Context-Aware Integration: Animates the clouds by continuously offsetting the UV coordinates of the noise functions. To simulate realistic atmospheric dynamics, it applies wind shear: high-altitude clouds (cirrus) move rapidly across the screen, while low-altitude clouds (stratus) drift slowly. The wind speed and direction are directly tied to the active weather preset, blowing fiercely during the Stormy Tempest and remaining nearly still during the Morning Mist.

<span style="color:red">ScreenSpaceCloudOptimizer.js</span>
Imports: `WebGLRenderer.js`, `Camera.js`, `Frustum.js`, `MathUtils.js`
Functions & Algorithms: `calculateCloudLOD()`, `fallbackTo2DBillboards()`, `cullOffScreenClouds()`, `optimizeFragmentShaderComplexity()`.
Context-Aware Integration: Ensures the sky system never bottlenecks the GPU. For clouds far in the distance (near the horizon), it bypasses the complex 3D noise calculations and renders simple, pre-baked 2D animated sprites or a simplified 2D noise function. It also dynamically reduces the number of fBm octaves evaluated in the fragment shader if the device's thermal state rises, maintaining a silky-smooth frame rate without visibly degrading the anime aesthetic.

<span style="color:red">SkyContextUniformBinder.js</span>
Imports: `Uniform.js`, `UniformsGroup.js`, `EventDispatcher.js`, `WebGLRenderer.js`
Functions & Algorithms: `registerSkyUniforms()`, `syncBiomeParameters()`, `updateTimeAndWeatherVectors()`, `batchShaderUploads()`.
Context-Aware Integration: The central communication node for the sky system. It listens to the global `ContextManager` for changes in biome, time of day, and weather. When a change occurs, it calculates the new target values for all sky and cloud uniforms (sun direction, cloud density, wind speed, color palettes) and pushes them to the GPU in a single, highly optimized batch, ensuring the sky, clouds, and terrain lighting remain perfectly synchronized.
