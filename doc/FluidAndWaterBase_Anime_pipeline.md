## BATCH 18: ARCHITECTURAL EXPANSION OF FLUIDANDWATERBASE.JS SUB-MODULES

To render breathtaking, dynamic water surfaces that perfectly capture the stylized anime aesthetic—complete with sharp wave crests, distinct color banding, and high-contrast specular glints—the `FluidAndWaterBase.js` module is decomposed into 10 highly specialized sub-files. These files manage the complex mathematics of fluid dynamics, screen-space effects, and context-aware rendering optimizations.

<span style="color:red">GerstnerWaveMathEngine.js</span>
Imports: `Vector3.js`, `Vector2.js`, `MathUtils.js`, `Clock.js`
Functions & Algorithms: `evaluateGerstnerSummation()`, `calculateVertexDisplacement()`, `computeWaveSteepness()`, `optimizeTrigonometricCalculations()`.
Context-Aware Integration: The core mathematical engine for vertex displacement. It sums multiple sine waves to create complex, intersecting wave patterns. The parameters are strictly bound to the active biome: it uses low-frequency, high-amplitude, and high-steepness waves for the crashing Coastal Oceanic preset, and high-frequency, low-amplitude, gentle ripples for the flowing River Canyon preset.

<span style="color:red">AnalyticalNormalCalculator.js</span>
Imports: `Vector3.js`, `MathUtils.js`, `ShaderMaterial.js`, `BufferAttribute.js`
Functions & Algorithms: `calculatePartialDerivatives()`, `computeAnalyticalNormals()`, `flattenNormalsForCelShading()`, `updateNormalAttributes()`.
Context-Aware Integration: Crucial for the anime water look. Instead of relying on bump maps, it calculates the exact surface normal mathematically from the Gerstner wave derivatives. It then passes these normals through a quantization function to create hard, faceted normal bands, ensuring the water reflects light in sharp, stylized steps rather than smooth, realistic gradients.

<span style="color:red">AnimeWaterFragmentAssembler.js</span>
Imports: `ShaderMaterial.js`, `ShaderChunk.js`, `String.js`, `WebGLCapabilities.js`
Functions & Algorithms: `assembleModularWaterCode()`, `injectBiomeSpecificMacros()`, `optimizeAlphaBlending()`, `compileAndCacheShaderPrograms()`.
Context-Aware Integration: Dynamically constructs the water fragment shader. It includes or excludes complex optical effects based on the camera distance and device performance. Distant ocean water uses a simplified, unlit shader with a static gradient, while foreground river water receives the full suite of refraction, reflection, and depth-based color calculations.

<span style="color:red">DepthGradientAndAbsorption.js</span>
Imports: `Color.js`, `MathUtils.js`, `Vector3.js`, `WebGLRenderer.js`
Functions & Algorithms: `sampleSceneDepthTexture()`, `calculateWaterDepth()`, `applyBeerLambertApproximation()`, `interpolateShallowToDeepColors()`.
Context-Aware Integration: Creates the iconic anime water color transitions. It samples the depth buffer to determine how deep the water is at each pixel. It then mathematically interpolates between the shallow color (transparent, vibrant cyan or emerald) and the deep color (opaque, dark navy or teal). The absorption rate is context-aware, allowing light to penetrate deeper in the tropical ocean than in the murky river.

<span style="color:red">StylizedSpecularHighlighter.js</span>
Imports: `Vector3.js`, `MathUtils.js`, `Color.js`, `ShaderMaterial.js`
Functions & Algorithms: `calculateHalfVector()`, `applyAnisotropicDistribution()`, `quantizeSpecularBands()`, `generateSunGlints()`.
Context-Aware Integration: Controls the sharp, blown-out highlights seen on anime water. Instead of a smooth Blinn-Phong specular term, it uses a strict `step` or narrow `smoothstep` function to create crisp, circular sun glints. The intensity and size of these glints are dynamically adjusted: massive, blinding highlights for the high-noon Ocean scene, and subtle, scattered sparkles for the shaded River Canyon.

<span style="color:red">PlanarReflectionManager.js</span>
Imports: `WebGLRenderTarget.js`, `PerspectiveCamera.js`, `Matrix4.js`, `Frustum.js`
Functions & Algorithms: `calculateReflectedViewMatrix()`, `renderReflectionPass()`, `applyScreenSpaceUVMapping()`, `manageReflectionLOD()`.
Context-Aware Integration: Handles the real-time mirroring of the sky and terrain on the water surface. It renders the scene from a mirrored camera perspective into a low-resolution render target. To maintain high FPS, it dynamically reduces the resolution of this reflection pass during complex weather events (like the Stormy Tempest) and applies a stylized blur to mask the lower resolution, fitting the anime aesthetic.

<span style="color:red">ShorelineFoamGenerator.js</span>
Imports: `ShaderMaterial.js`, `Vector3.js`, `MathUtils.js`, `Color.js`
Functions & Algorithms: `calculateShorelineProximity()`, `evaluateWaveVelocity()`, `generateProceduralFoamNoise()`, `blendFoamColorWithWater()`.
Context-Aware Integration: Creates the white, frothy edges where water meets land or where waves crest. It uses a combination of depth thresholding (for shorelines) and vertex velocity (for wave crests) to drive a procedural noise function. The foam is rendered purely through color manipulation and alpha blending, avoiding heavy particle systems for distant water to save CPU/GPU overhead.

<span style="color:red">RefractionAndDistortionSystem.js</span>
Imports: `Vector2.js`, `ShaderMaterial.js`, `WebGLRenderer.js`, `MathUtils.js`
Functions & Algorithms: `calculateScreenSpaceUVOffset()`, `applyNormalMapDistortion()`, `sampleRefractedColor()`, `simulateCausticPatterns()`.
Context-Aware Integration: Distorts the view of the riverbed or submerged rocks beneath the surface. It uses the analytically calculated water normals to offset the screen-space UV coordinates when sampling the scene color behind the water. The distortion strength is context-aware: high and chaotic for the shallow, fast-moving river, and low and smooth for the deep, calm ocean.

<span style="color:red">WaterContextUniformBinder.js</span>
Imports: `Uniform.js`, `UniformsGroup.js`, `EventDispatcher.js`, `Clock.js`
Functions & Algorithms: `registerWaterUniforms()`, `syncBiomeParameters()`, `updateTimeAndWindVectors()`, `batchShaderUploads()`.
Context-Aware Integration: The central communication node for the water system. It listens to the global `ContextManager` for changes in weather and time. When a storm hits, it instantly updates the wave amplitude, foam density, and color saturation uniforms across all water materials in a single batch, ensuring the ocean reacts immediately and cohesively to the environmental shift.

<span style="color:red">FluidPerformanceOptimizer.js</span>
Imports: `Frustum.js`, `Camera.js`, `MathUtils.js`, `WebGLRenderer.js`
Functions & Algorithms: `calculateScreenSpaceTessellation()`, `cullOffScreenWaterPatches()`, `switchToFlatShadedLOD()`, `optimizeDrawCallOrder()`.
Context-Aware Integration: Ensures the water system never bottlenecks the GPU. It dynamically adjusts the tessellation level of the water plane based on distance. Foreground water is highly tessellated to support detailed Gerstner waves and foam, while distant water is reduced to a single, flat-shaded quad with a scrolling normal map texture, maintaining a locked 60 FPS even on vast ocean scenes.
