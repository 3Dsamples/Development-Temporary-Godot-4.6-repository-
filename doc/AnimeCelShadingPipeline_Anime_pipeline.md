## BATCH 11: ARCHITECTURAL EXPANSION OF ANIMECELSHADINGPIPELINE.JS SUB-MODULES

To achieve a flawless, high-performance anime aesthetic that dynamically adapts to every biome and weather condition, the `AnimeCelShadingPipeline.js` module is decomposed into 10 specialized sub-files. These files manage the GLSL shader generation, mathematical quantization, and context-aware material routing without relying on heavy texture sampling.

<span style="color:red">CelVertexShaderGenerator.js</span>
Imports: `ShaderChunk.js`, `Vector3.js`, `Matrix4.js`, `WebGLCapabilities.js`
Functions & Algorithms: `generateWorldSpaceNormals()`, `calculateViewAndLightVectors()`, `packAttributesForPerformance()`, `injectMorphTargetSupport()`.
Context-Aware Integration: Dynamically adjusts the floating-point precision (`highp`, `mediump`, `lowp`) of the vertex shader based on the active device tier. For mobile devices in the vast Desert preset, it downgrades to `mediump` to maximize vertex throughput, while maintaining `highp` for the intricate House scene to prevent lighting artifacts on small architectural details.

<span style="color:red">CelFragmentShaderAssembler.js</span>
Imports: `ShaderMaterial.js`, `ShaderChunk.js`, `String.js`
Functions & Algorithms: `assembleModularShaderCode()`, `injectBiomeSpecificMacros()`, `optimizeDeadCodeElimination()`, `compileAndCacheShaderPrograms()`.
Context-Aware Integration: Acts as a dynamic compiler. Instead of compiling a massive monolithic shader, it concatenates only the GLSL chunks required for the current context. If the Stormy preset is active, it strips out the complex rim-lighting and specular chunks to save GPU instruction cycles, ensuring maximum FPS during heavy particle rendering.

<span style="color:red">DiffuseQuantizationEngine.js</span>
Imports: `MathUtils.js`, `Vector3.js`, `Color.js`
Functions & Algorithms: `calculateDotProductNL()`, `applySmoothstepThresholds()`, `generateMultiToneBands()`, `interpolateShadowBoundaries()`.
Context-Aware Integration: The core of the anime look. It dynamically adjusts the `smoothstep` width based on the weather context. During the Clear Coastal preset, it uses a near-zero width for razor-sharp, 2-tone cel shadows. During the Morning Mist preset, it widens the `smoothstep` gradient to create soft, blended 3-tone shadows that match the diffused atmospheric mood.

<span style="color:red">StylizedRimLightingSystem.js</span>
Imports: `Vector3.js`, `Color.js`, `MathUtils.js`
Functions & Algorithms: `calculateFresnelApproximation()`, `applyPowerCurveExponent()`, `maskRimByLightDirection()`, `blendRimColorWithBase()`.
Context-Aware Integration: Modulates the rim light intensity and color based on the active biome. For the Alpine Cliff preset, it injects a sharp, high-intensity cyan rim to separate the dark rock from the bright sky. For the Golden Hour preset, it shifts the rim color to a warm, glowing gold and lowers the power exponent to create a softer, broader halo around the vegetation.

<span style="color:red">AnimeSpecularHighlighter.js</span>
Imports: `Vector3.js`, `MathUtils.js`, `Color.js`
Functions & Algorithms: `calculateHalfVector()`, `applyAnisotropicDistribution()`, `shapeSpecularHighlights()`, `clampSpecularIntensity()`.
Context-Aware Integration: Controls the presence and shape of specular reflections. In the Arid Desert preset, it completely disables specular highlights on the sand to maintain a matte, dry look. In the Coastal Oceanic preset, it activates sharp, circular, high-intensity speculars on the water and wet rocks, mimicking the stylized, blown-out highlights typical of anime water rendering.

<span style="color:red">ContextAwareColorBanding.js</span>
Imports: `Color.js`, `MathUtils.js`, `Vector3.js`
Functions & Algorithms: `convertRGBtoHSV()`, `shiftHueBasedOnLightAngle()`, `adjustSaturationInShadows()`, `convertHSVtoRGB()`.
Context-Aware Integration: Prevents shadows from looking muddy by dynamically shifting the hue of the shadow bands. In the River Canyon preset, it shifts shadow hues towards cool purples and blues to simulate bounced sky light. In the Temperate Rural preset, it shifts shadow hues towards warm magentas to simulate the rich, earthy bounce light of the garden.

<span style="color:red">OutlineRenderingController.js</span>
Imports: `Mesh.js`, `ShaderMaterial.js`, `Vector3.js`, `Color.js`
Functions & Algorithms: `setupInvertedHullGeometry()`, `calculateBackfaceExtrusion()`, `applyVariableOutlineThickness()`, `assignContextualOutlineColors()`.
Context-Aware Integration: Manages the iconic anime black outlines using the inverted hull method. It dynamically adjusts the extrusion depth and outline color based on the object's context. The ocean rocks receive a thick, dark blue outline, while the delicate house vegetation receives a thin, dark green outline, ensuring visual hierarchy and readability without manual artist intervention.

<span style="color:red">ShadowTintingAndBounce.js</span>
Imports: `Color.js`, `Vector3.js`, `HemisphereLight.js`
Functions & Algorithms: `sampleHemisphereSkyColor()`, `sampleHemisphereGroundColor()`, `calculateAmbientOcclusionFactor()`, `blendBounceLightIntoShadows()`.
Context-Aware Integration: Replaces flat black shadows with rich, colored ambient occlusion. It samples the global `HemisphereLight` colors and injects them into the shadow bands. For the Polar Glacial preset, it tints the shadows with a crisp lavender-blue to simulate light scattering through snow, vastly improving the color composition without adding extra light sources.

<span style="color:red">ShaderUniformBinder.js</span>
Imports: `Uniform.js`, `UniformsGroup.js`, `WebGLRenderer.js`
Functions & Algorithms: `mapJavaScriptVariablesToGLSL()`, `batchUniformUpdates()`, `implementDirtyFlagChecking()`, `optimizeUniformBufferObjects()`.
Context-Aware Integration: Ensures the highest possible CPU-to-GPU transfer speed. It groups all anime pipeline uniforms (time, sun direction, mood factors) into WebGL Uniform Buffer Objects (UBOs). It only pushes updates to the GPU when a specific uniform's value actually changes, eliminating redundant state changes and freeing up CPU cycles for the procedural geometry generators.

<span style="color:red">MaterialVariantManager.js</span>
Imports: `ShaderMaterial.js`, `Material.js`, `Object3D.js`
Functions & Algorithms: `registerMaterialPresets()`, `dynamicallySwapShaderChunks()`, `transitionBetweenVariants()`, `cacheCompiledPrograms()`.
Context-Aware Integration: Allows a single mesh to change its shading behavior based on its environment. When a rock transitions from the dry Desert zone into the River water zone, this manager seamlessly swaps the material's fragment shader from the "Matte Cel" variant to the "Wet Cel with Specular" variant in real-time, maintaining perfect visual continuity and context-awareness.

