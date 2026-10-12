## BATCH 10: ARCHITECTURAL EXPANSION OF COREENGINEINIT.JS SUB-MODULES

To achieve the highest possible performance and maintain a strictly context-aware, interactive real-time anime pipeline, the `CoreEngineInit.js` module is decomposed into 10 highly specialized, interconnected sub-files. These files manage the foundational WebGL context, memory allocation, state optimization, and global event routing.

<span style="color:red">WebGLContextManager.js</span>
Imports: `WebGLRenderer.js`, `WebGLCapabilities.js`, `Vector2.js`
Functions & Algorithms: `createHighPerformanceContext()`, `handleContextLossAndRestoration()`, `calculateAdaptivePixelRatio()`, `configureWebGL2Extensions()`.
Context-Aware Integration: Monitors the device's thermal and performance tier. If the GPU is identified as mobile or integrated, it automatically restricts the WebGL context to WebGL1 or disables advanced extensions like floating-point textures to prevent crashes, ensuring the anime pipeline remains stable across all hardware.

<span style="color:red">RendererStateOptimizer.js</span>
Imports: `WebGLState.js`, `WebGLProperties.js`, `RenderList.js`, `SortUtils.js`
Functions & Algorithms: `minimizeStateChanges()`, `batchDrawCallsByMaterial()`, `cacheWebGLUniforms()`, `optimizeRenderOrder()`.
Context-Aware Integration: Intercepts the render list before execution. It dynamically re-sorts opaque and transparent objects based on the active biome's complexity. For the dense Tropical Rainforest preset, it aggressively batches vegetation draw calls, whereas for the open Desert preset, it prioritizes shadow map rendering order to maximize cascade efficiency.

<span style="color:red">SceneGraphContextHub.js</span>
Imports: `Scene.js`, `EventDispatcher.js`, `Object3D.js`
Functions & Algorithms: `initializeGlobalEventBus()`, `broadcastBiomeTransition()`, `subscribeToMoodChanges()`, `manageSceneGraphHierarchy()`.
Context-Aware Integration: Acts as the central nervous system. When a climatic change is triggered (e.g., transitioning from Clear to Stormy), this hub emits a `CONTEXT_UPDATE` event. All subscribed modules (Lighting, Fog, Water) receive the event simultaneously and begin interpolating their parameters, ensuring perfectly synchronized, glitch-free environmental transitions.

<span style="color:red">CameraFrustumController.js</span>
Imports: `PerspectiveCamera.js`, `Matrix4.js`, `Frustum.js`, `MathUtils.js`
Functions & Algorithms: `calculateDynamicNearFarPlanes()`, `adjustContextAwareFOV()`, `preventZFighting()`, `updateProjectionMatrixOptimized()`.
Context-Aware Integration: Continuously analyzes the bounding box of the active scene elements. For the vast River Canyon, it expands the far plane and tightens the near plane to maximize depth buffer precision. For the close-up House scene, it restricts the far plane to prevent z-fighting on the intricate architectural details and window frames.

<span style="color:red">AnimationLoopScheduler.js</span>
Imports: `Clock.js`, `MathUtils.js`, `RequestAnimationFrame.js`
Functions & Algorithms: `setupOptimizedRequestLoop()`, `calculateSmoothedDeltaTime()`, `separatePhysicsAndRenderTicks()`, `manageFramePacing()`.
Context-Aware Integration: Implements a dual-tick system. The physics and particle systems (like Ocean Foam and Desert Wind) run on a fixed time step for deterministic behavior, while the rendering and camera updates run on a variable time step. This prevents animation stutter during heavy GPU loads and ensures the anime cel-shading remains perfectly smooth.

<span style="color:red">PerformanceMonitorAndAdaptor.js</span>
Imports: `Clock.js`, `WebGLRenderer.js`, `Stats.js`
Functions & Algorithms: `trackFrameTimes()`, `detectThermalThrottling()`, `dynamicallyScaleResolution()`, `adjustShadowMapQuality()`.
Context-Aware Integration: Continuously profiles the main thread and GPU execution time. If the frame time exceeds the 16ms threshold for three consecutive frames, it automatically downgrades the shadow map resolution from 2048x2048 to 1024x1024 and reduces the pixel ratio, guaranteeing a locked 60 FPS even during complex climatic transitions.

<span style="color:red">MemoryAndResourceAllocator.js</span>
Imports: `ObjectPool.js`, `Vector3.js`, `Matrix4.js`, `Color.js`
Functions & Algorithms: `initializeObjectPools()`, `allocateTransientVectors()`, `cacheGeometryAndMaterials()`, `preventGarbageCollectionStutter()`.
Context-Aware Integration: Pre-allocates massive pools of `Vector3`, `Matrix4`, and `Color` objects used by the procedural generators. When the Instancing system calculates the position of 10,000 rocks, it borrows vectors from the pool and returns them immediately, completely eliminating JavaScript garbage collection spikes that cause frame drops.

<span style="color:red">PostProcessingPipelineInit.js</span>
Imports: `WebGLRenderTarget.js`, `ShaderMaterial.js`, `ColorSpace.js`
Functions & Algorithms: `setupMultiRenderTarget()`, `configureACESFilmicToneMapping()`, `initializeLinearToSRGBConversion()`, `allocateBloomBuffers()`.
Context-Aware Integration: Sets up the render targets required for the anime color pipeline. It configures the tone mapping curve based on the mood preset; for the Golden Hour preset, it shifts the tone mapping curve to preserve warm highlights, while for the Stormy preset, it compresses the highlights to maintain a dark, oppressive contrast.

<span style="color:red">InputAndInteractionRouter.js</span>
Imports: `Raycaster.js`, `Vector2.js`, `Camera.js`, `EventDispatcher.js`
Functions & Algorithms: `initializeRaycastingSystem()`, `mapPointerToNormalizedDeviceCoordinates()`, `dispatchContextAwareInteractions()`, `optimizeIntersectionTests()`.
Context-Aware Integration: Routes user input to the 3D scene. It dynamically adjusts the raycaster's precision and target layers based on the context. In the House scene, it enables high-precision raycasting for interacting with doors and windows. In the vast Desert scene, it disables precise raycasting on distant dunes to save CPU cycles, only enabling it for foreground rocks.

<span style="color:red">GlobalUniformRegistry.js</span>
Imports: `Uniform.js`, `UniformsGroup.js`, `Clock.js`
Functions & Algorithms: `registerGlobalShaderUniforms()`, `automateDirtyChecking()`, `updateTimeAndWeatherUniforms()`, `syncUniformsAcrossMaterials()`.
Context-Aware Integration: Maintains a centralized dictionary of all global uniforms (e.g., `uTime`, `uSunDirection`, `uMoodFactor`). Instead of updating uniforms per material, it updates them once per frame in this registry. All anime cel-shading and water shaders reference this registry, ensuring that when the sun moves, every single object in the scene updates its lighting simultaneously with zero redundant CPU overhead.

