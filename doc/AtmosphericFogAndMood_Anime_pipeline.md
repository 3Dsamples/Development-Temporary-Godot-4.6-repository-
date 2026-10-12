## BATCH 13: ARCHITECTURAL EXPANSION OF ATMOSPHERICFOGANDMOOD.JS SUB-MODULES

To achieve a deeply immersive, context-aware atmospheric system that seamlessly transitions between biomes and weather conditions while maintaining the stylized anime aesthetic, the `AtmosphericFogAndMood.js` module is decomposed into 10 highly specialized sub-files. These files manage the complex mathematical modeling of light scattering, depth-based color interpolation, and global mood synchronization.

<span style="color:red">ExponentialFogDensityCalculator.js</span>
Imports: `FogExp2.js`, `MathUtils.js`, `Vector3.js`, `Camera.js`
Functions & Algorithms: `calculateSquaredDistance()`, `evaluateBeerLambertLaw()`, `optimizeExponentialDecay()`, `computePerPixelFogFactor()`.
Context-Aware Integration: Implements the core mathematical foundation of the fog system using the equation `factor = 1.0 - exp(-density * distance^2)`. To maximize GPU performance, it calculates the squared distance in the vertex shader and passes it to the fragment shader, avoiding expensive square root operations. The density parameter is dynamically bound to the active climatic preset, ensuring the dense, opaque fog of the Morning Mist and the thin, crisp haze of the Alpine Cliff are mathematically distinct.

<span style="color:red">DepthBasedColorBlender.js</span>
Imports: `Color.js`, `Vector3.js`, `ShaderMaterial.js`, `MathUtils.js`
Functions & Algorithms: `linearInterpolateColors()`, `applyFogFactorToRGB()`, `preventColorClipping()`, `blendObjectAndAtmosphere()`.
Context-Aware Integration: Handles the final pixel composition by linearly interpolating (lerp) between the object's shaded color and the global fog color using the calculated fog factor. It includes a context-aware saturation preservation algorithm, ensuring that as objects fade into the distance (like the distant mountains in the Desert scene), they do not become muddy or desaturated, but instead retain a vibrant, stylized hue that matches the anime reference images.

<span style="color:red">AtmosphericScatteringSimulator.js</span>
Imports: `Vector3.js`, `Color.js`, `MathUtils.js`, `Camera.js`
Functions & Algorithms: `approximateRayleighScattering()`, `simulateMieScattering()`, `calculateHorizonColorShift()`, `modulateScatteringBySunAngle()`.
Context-Aware Integration: Enhances the basic fog model by simulating physical light scattering. It calculates the angle between the view vector and the sun direction to apply a subtle, forward-scattering glow (Mie scattering) around the sun, particularly visible in the Golden Hour and Coastal Oceanic presets. This adds a layer of physical realism to the anime style, making the sky and horizon feel vast and luminous.

<span style="color:red">MoodTransitionInterpolater.js</span>
Imports: `Color.js`, `Vector3.js`, `MathUtils.js`, `EventDispatcher.js`
Functions & Algorithms: `slerpMoodVectors()`, `interpolateFogDensity()`, `smoothStepTransition()`, `cacheTransitionStates()`.
Context-Aware Integration: Manages the seamless transition between different climatic presets (e.g., from Clear to Stormy). Instead of abrupt changes, it uses spherical linear interpolation (slerp) for color vectors and smoothstep functions for density parameters. This ensures that when the weather changes, the fog color shifts smoothly from a pale blue to a dark slate grey over a configurable duration, preventing visual popping and maintaining a cinematic feel.

<span style="color:red">HeightBasedFogModulator.js</span>
Imports: `Vector3.js`, `MathUtils.js`, `FogExp2.js`, `Camera.js`
Functions & Algorithms: `calculateWorldHeightDelta()`, `applyVerticalDensityGradient()`, `modulateFogByAltitude()`, `optimizeHeightUniforms()`.
Context-Aware Integration: Crucial for vertical biomes like the Alpine Cliff and River Canyon. It modifies the base fog density based on the world Y-coordinate of the pixel. For the cliff scene, it increases the fog density at lower altitudes to simulate a sea of clouds or valley mist, while keeping the upper peaks crisp and clear. This is achieved via a simple linear gradient function injected into the fragment shader, adding immense depth to the scene with minimal computational cost.

<span style="color:red">VolumetricLightShaftIntegrator.js</span>
Imports: `ShaderMaterial.js`, `Vector3.js`, `Color.js`, `MathUtils.js`
Functions & Algorithms: `calculateCrepuscularRayFactor()`, `sampleShadowMapForOcclusion()`, `applyRadialBlurApproximation()`, `blendLightShafts()`.
Context-Aware Integration: Simulates god rays (volumetric light shafts) piercing through the atmosphere, essential for the Tropical Rainforest and Morning Mist presets. It uses a simplified, screen-space radial blur technique originating from the sun's position, masked by the scene's depth and shadow maps. This creates the illusion of light scattering through fog and canopy gaps, dramatically enhancing the mood and color composition without the heavy cost of true volumetric rendering.

<span style="color:red">ContextAwareFogColorPicker.js</span>
Imports: `Color.js`, `Vector3.js`, `MathUtils.js`, `HemisphereLight.js`
Functions & Algorithms: `sampleSkyAndGroundColors()`, `calculateWeightedAverageColor()`, `applyTimeOfDayShift()`, `updateGlobalFogUniforms()`.
Context-Aware Integration: Dynamically determines the optimal fog color based on the active biome and time of day. It samples the `HemisphereLight` colors and applies a context-aware weighting function. For the Polar Glacial preset, it selects a crisp lavender-blue; for the Arid Desert, it selects a warm, dusty ochre. This ensures the fog always acts as a cohesive color bridge between the sky and the ground, unifying the scene's palette.

<span style="color:red">FogOcclusionAndCullingOptimizer.js</span>
Imports: `Frustum.js`, `Camera.js`, `Object3D.js`, `MathUtils.js`
Functions & Algorithms: `calculateFogCullingDistance()`, `updateBoundingSpheres()`, `aggressivelyCullDistantMeshes()`, `optimizeCPUTraversal()`.
Context-Aware Integration: A vital CPU-side optimization. It calculates the exact distance at which the fog factor reaches 1.0 (complete opacity) based on the current density. It then uses this distance to create a tight bounding sphere around the camera. Any object whose bounding sphere falls entirely outside this "visible fog radius" is automatically culled from the render list, drastically reducing draw calls in dense environments like the Desert or Ocean scenes.

<span style="color:red">AnimeStyleFogQuantizer.js</span>
Imports: `ShaderMaterial.js`, `MathUtils.js`, `Color.js`, `Vector3.js`
Functions & Algorithms: `applyStepFunctionToFogFactor()`, `generateDistinctFogBands()`, `quantizeAtmosphericDepth()`, `alignWithCelShading()`.
Context-Aware Integration: Adapts the smooth, physical fog model to the stylized anime aesthetic. Instead of a continuous gradient, it passes the fog factor through a quantization function (e.g., a multi-step smoothstep). This creates distinct, hard-edged bands of atmospheric depth, mimicking the background art style of classic anime where distant mountains are rendered in flat, layered silhouettes of color rather than smooth gradients.

<span style="color:red">GlobalMoodEventDispatcher.js</span>
Imports: `EventDispatcher.js`, `Clock.js`, `Vector3.js`, `Color.js`
Functions & Algorithms: `emitMoodChangeEvents()`, `synchronizeSubsystemUpdates()`, `manageTransitionTimers()`, `broadcastGlobalUniformChanges()`.
Context-Aware Integration: The central communication hub for the atmospheric system. When a mood transition is initiated, this dispatcher emits a standardized event containing the target fog color, density, and transition duration. All subscribed systems (Water, Lighting, Vegetation) receive this event and begin their own synchronized transitions, ensuring that the water color shifts, the lights dim, and the fog thickens in perfect unison, creating a flawless, immersive climatic change.

