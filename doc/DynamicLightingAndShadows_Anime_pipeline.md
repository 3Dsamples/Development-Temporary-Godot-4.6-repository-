## BATCH 12: ARCHITECTURAL EXPANSION OF DYNAMICLIGHTINGANDSHADOWS.JS SUB-MODULES

To achieve physically plausible yet highly stylized anime lighting with maximum GPU efficiency, the `DynamicLightingAndShadows.js` module is decomposed into 10 specialized sub-files. These files manage the complex mathematics of cascaded shadow mapping, dynamic light positioning, and the precise quantization of shadow boundaries required for the anime aesthetic.

<span style="color:red">DynamicSunPositioning.js</span>
Imports: `DirectionalLight.js`, `Vector3.js`, `Color.js`, `MathUtils.js`
Functions & Algorithms: `calculateSolarElevationAndAzimuth()`, `interpolateColorTemperature()`, `modulateIntensityByWeather()`, `updateGlobalLightUniforms()`.
Context-Aware Integration: Drives the primary light source based on a normalized time-of-day float and the active climatic preset. During the Golden Hour preset, it mathematically lowers the solar elevation angle and shifts the color temperature to a deep, saturated orange (RGB: 1.0, 0.4, 0.1). During the Stormy Tempest preset, it drastically reduces the intensity and shifts the color to a cold, desaturated grey-blue, instantly altering the mood of the entire scene.

<span style="color:red">CascadedShadowMapAllocator.js</span>
Imports: `WebGLRenderTarget.js`, `Matrix4.js`, `Vector3.js`, `Frustum.js`
Functions & Algorithms: `partitionCameraFrustum()`, `allocateShadowMapTextures()`, `calculateOrthographicBounds()`, `manageRenderToTextureMemory()`.
Context-Aware Integration: Splits the camera's view frustum into four distinct depth cascades. It dynamically allocates texture memory based on the active biome. For the close-up House scene, it allocates higher resolution to the first cascade to capture sharp shadows on the roof tiles. For the vast Desert scene, it expands the bounds of the fourth cascade to ensure distant mountains cast accurate shadows without exhausting VRAM.

<span style="color:red">AnimeShadowQuantizer.js</span>
Imports: `ShaderMaterial.js`, `MathUtils.js`, `Vector3.js`, `Color.js`
Functions & Algorithms: `sampleShadowMapDepth()`, `applyStepFunctionThreshold()`, `calculateHardEdgeTransition()`, `injectQuantizationUniforms()`.
Context-Aware Integration: The mathematical core of the anime shadow look. Instead of standard Percentage-Closer Filtering (PCF) which creates soft, realistic shadows, this module applies a strict `step` or narrow `smoothstep` function to the shadow map depth comparison. The threshold width is context-aware: it is set to near-zero for the crisp, hard shadows of the Coastal Oceanic preset, and slightly widened for the softer, diffused shadows of the Morning Mist preset.

<span style="color:red">HemisphereGradientSolver.js</span>
Imports: `HemisphereLight.js`, `Color.js`, `Vector3.js`, `MathUtils.js`
Functions & Algorithms: `calculateSkyGroundDotProduct()`, `interpolateGradientColors()`, `applySphericalHarmonicsApproximation()`, `updateAmbientBounceUniforms()`.
Context-Aware Integration: Computes the ambient bounce lighting by interpolating between the sky color and the ground color based on the surface normal's Y-axis orientation. It dynamically updates these colors to match the active preset, ensuring that the underside of the cliff rocks receives a cool blue bounce from the ocean, while the desert dunes receive a warm ochre bounce from the sand, enriching the color composition without extra draw calls.

<span style="color:red">ShadowBiasAndAcnePreventer.js</span>
Imports: `WebGLShadowMap.js`, `Matrix4.js`, `Vector3.js`, `MathUtils.js`
Functions & Algorithms: `calculateNormalOffsetBias()`, `applySlopeScaleDepthBias()`, `detectShadowAcne()`, `autoTuneBiasParameters()`.
Context-Aware Integration: Crucial for flat-shaded anime geometry. Large, flat surfaces like the house walls or the cliff faces are highly susceptible to shadow acne (striping artifacts). This module mathematically calculates the optimal depth bias based on the angle between the light direction and the surface normal, automatically injecting a slight offset into the depth comparison to eliminate artifacts while preventing "peter-panning" (shadows detaching from objects).

<span style="color:red">FrustumCulledLightCaster.js</span>
Imports: `Frustum.js`, `Matrix4.js`, `WebGLRenderer.js`, `RenderList.js`
Functions & Algorithms: `calculateLightSpaceFrustum()`, `cullInvisibleMeshes()`, `optimizeShadowPassDrawCalls()`, `batchShadowCasters()`.
Context-Aware Integration: Drastically reduces CPU overhead during the shadow map rendering pass. It calculates the exact bounding volume of the light's orthographic frustum and aggressively culls any geometry (like distant ocean waves or far desert rocks) that does not intersect it. This ensures that the GPU only processes shadows for objects that actually contribute to the final image, maintaining high FPS in complex scenes.

<span style="color:red">DynamicShadowResolutionScaler.js</span>
Imports: `WebGLRenderer.js`, `WebGLRenderTarget.js`, `Clock.js`, `MathUtils.js`
Functions & Algorithms: `monitorFrameExecutionTime()`, `calculateOptimalShadowResolution()`, `resizeShadowMapTextures()`, `rebindRenderTargets()`.
Context-Aware Integration: A vital performance safeguard. If the `PerformanceMonitor` detects thermal throttling or a sustained FPS drop, this module dynamically downgrades the shadow map resolution (e.g., from 2048x2048 to 1024x1024) in real-time. Once the scene complexity decreases (e.g., the camera looks away from the dense forest), it smoothly scales the resolution back up, guaranteeing a locked frame rate.

<span style="color:red">AmbientOcclusionColorBlender.js</span>
Imports: `Color.js`, `Vector3.js`, `ShaderMaterial.js`, `MathUtils.js`
Functions & Algorithms: `calculateGeometricAO()`, `sampleEnvironmentColor()`, `blendOcclusionTint()`, `preventMuddyShadows()`.
Context-Aware Integration: Enhances the anime aesthetic by replacing flat black shadows with rich, colored ambient occlusion. It calculates the occlusion factor based on the geometry's curvature and blends it with the `HemisphereLight`'s ground color. This ensures that the deep crevices of the canyon walls take on a deep, rich terracotta hue rather than a lifeless black, vastly improving the visual depth and color variety.

<span style="color:red">PenumbraAndSoftEdgeController.js</span>
Imports: `ShaderMaterial.js`, `MathUtils.js`, `Vector3.js`, `Color.js`
Functions & Algorithms: `calculatePenumbraWidth()`, `applyGaussianBlurToShadowEdge()`, `modulateSoftnessByDistance()`, `updateShadowFilterUniforms()`.
Context-Aware Integration: While anime shadows are typically hard, this module provides context-aware control over the shadow edge softness. For the harsh, direct sunlight of the Arid Desert, the penumbra width is set to zero. For the overcast, diffused light of the Stormy Tempest, it applies a subtle, mathematically precise Gaussian blur to the shadow edges, perfectly matching the atmospheric scattering and mood of the weather preset.

<span style="color:red">LightSpaceMatrixOptimizer.js</span>
Imports: `Matrix4.js`, `Vector3.js`, `Frustum.js`, `MathUtils.js`
Functions & Algorithms: `computeViewAlignedLightMatrix()`, `minimizeLightFrustumVolume()`, `maximizeTexelDensity()`, `updateLightProjectionMatrix()`.
Context-Aware Integration: Optimizes the orthographic projection matrix of the sun. Instead of using a massive, static bounding box for the entire world, it tightly wraps the light's frustum around the camera's current view frustum. This mathematical optimization maximizes the shadow map texel density exactly where the player is looking, ensuring that the shadows on the house roof or the foreground rocks are razor-sharp and highly detailed.

