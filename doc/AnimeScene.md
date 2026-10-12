## BATCH 8: MODULAR ARCHITECTURE AND SHARED CORE SYSTEMS

To achieve the highest performance and maintainability, the 50 files are reorganized using a strict modular architecture. We extract the **Shared Core Modules** (10 files) that contain the fundamental algorithms, shaders, and systems used across all scenes. The remaining 40 files are **Scene-Specific Modules** (10 per image) that act as configurations, extensions, and specialized generators inheriting from the core.

### SHARED CORE MODULES (The "Join Together" Group)

<span style="color:red">CoreEngineInit.js</span>
Imports: `Three.js`, `WebGLRenderer.js`, `Scene.js`, `PerspectiveCamera.js`
Functions & Algorithms: `initHighPerformanceRenderer()`, `initSceneGraph()`, `initCameraMatrix()`, `setupOptimizedAnimationLoop()`.

<span style="color:red">AnimeCelShadingPipeline.js</span>
Imports: `ShaderMaterial.js`, `Color.js`, `Vector3.js`, `MathUtils.js`
Functions & Algorithms: `createCelVertexShader()`, `createCelFragmentShader()`, `quantizeDiffuseLighting()`, `applyStylizedRimLighting()`.

<span style="color:red">DynamicLightingAndShadows.js</span>
Imports: `DirectionalLight.js`, `HemisphereLight.js`, `Color.js`, `Vector3.js`, `WebGLShadowMap.js`
Functions & Algorithms: `setupDynamicSun()`, `calculateCascadedShadowMaps()`, `applyAnimeShadowBands()`, `updateHemisphereGradients()`.

<span style="color:red">AtmosphericFogAndMood.js</span>
Imports: `FogExp2.js`, `Color.js`, `Vector3.js`, `MathUtils.js`
Functions & Algorithms: `setupExponentialFog()`, `calculateDepthFading()`, `blendAtmosphericColors()`, `updateMoodParameters()`.

<span style="color:red">ProceduralSkyAndClouds.js</span>
Imports: `SphereGeometry.js`, `ShaderMaterial.js`, `Color.js`, `Vector3.js`, `MathUtils.js`
Functions & Algorithms: `createSkyDome()`, `generateSkyGradient()`, `renderProceduralCloudNoise()`, `updateCloudBillboards()`.

<span style="color:red">InstancingAndColorVariation.js</span>
Imports: `InstancedMesh.js`, `Matrix4.js`, `Color.js`, `Frustum.js`
Functions & Algorithms: `initInstancedMeshes()`, `updateInstanceMatrices()`, `randomizeInstanceColors()`, `cullDistantInstances()`.

<span style="color:red">BaseProceduralGeometry.js</span>
Imports: `BufferGeometry.js`, `Vector3.js`, `MathUtils.js`, `IcosahedronGeometry.js`, `PlaneGeometry.js`
Functions & Algorithms: `displaceVerticesWithNoise()`, `computeFlatNormals()`, `applySignedDistanceFields()`, `generateBaseTerrain()`.

<span style="color:red">VegetationAndSSSSystem.js</span>
Imports: `ShaderMaterial.js`, `Color.js`, `Vector3.js`, `MathUtils.js`
Functions & Algorithms: `createVegetationVertexShader()`, `createVegetationFragmentShader()`, `applyCheapSSSWrap()`, `calculateBacklightColorShift()`.

<span style="color:red">FluidAndWaterBase.js</span>
Imports: `ShaderMaterial.js`, `Vector2.js`, `Vector3.js`, `Color.js`, `WebGLRenderTarget.js`
Functions & Algorithms: `createWaterVertexShader()`, `createWaterFragmentShader()`, `calculateGerstnerWaves()`, `applyDepthGradient()`.

<span style="color:red">ParticleAndWindSystem.js</span>
Imports: `PointsMaterial.js`, `BufferGeometry.js`, `Vector3.js`, `MathUtils.js`
Functions & Algorithms: `initParticleSystems()`, `updateParticlePositions()`, `calculateWindVectors()`, `blendParticleOpacity()`.

---

### IMAGE 1: COASTAL OCEANIC (Ocean with Rocks)

<span style="color:red">OceanSceneConfig.js</span>
Imports: `CoreEngineInit.js`, `AtmosphericFogAndMood.js`
Functions & Algorithms: `configureOceanCamera()`, `setCoastalFogDensity()`, `applyOceanicMoodColors()`.

<span style="color:red">CoastalRockGenerator.js</span>
Imports: `BaseProceduralGeometry.js`, `InstancingAndColorVariation.js`
Functions & Algorithms: `generateJaggedCoastalRocks()`, `applyVertexDisplacement()`, `configureRockColorPalette()`.

<span style="color:red">OceanFoamAndShoreline.js</span>
Imports: `FluidAndWaterBase.js`, `ParticleAndWindSystem.js`
Functions & Algorithms: `calculateShorelineProximity()`, `generateFoamParticles()`, `blendFoamColorWithWater()`.

<span style="color:red">OceanicSkyPreset.js</span>
Imports: `ProceduralSkyAndClouds.js`
Functions & Algorithms: `setAzureToCyanGradient()`, `configureSparseWorleyClouds()`, `updateSkyUniforms()`.

<span style="color:red">OceanReflectionPass.js</span>
Imports: `FluidAndWaterBase.js`, `DynamicLightingAndShadows.js`
Functions & Algorithms: `initPlanarReflectionTarget()`, `renderReflectionPass()`, `calculateFresnelEffect()`.

<span style="color:red">CoastalLightingSetup.js</span>
Imports: `DynamicLightingAndShadows.js`
Functions & Algorithms: `configureHighZenithSun()`, `setSharpShadowThresholds()`, `applyWaterSpecularHighlights()`.

<span style="color:red">RockInstancingConfig.js</span>
Imports: `InstancingAndColorVariation.js`, `CoastalRockGenerator.js`
Functions & Algorithms: `batchCoastalRocks()`, `randomizeRockRotations()`, `applyAmbientOcclusionColors()`.

<span style="color:red">OceanAtmosphereConfig.js</span>
Imports: `AtmosphericFogAndMood.js`
Functions & Algorithms: `setHorizonBlendColors()`, `configureDistanceFading()`, `updateAtmosphericScattering()`.

<span style="color:red">WaterDepthBlending.js</span>
Imports: `FluidAndWaterBase.js`
Functions & Algorithms: `mapDepthToCyanNavyGradient()`, `calculateSubsurfaceAbsorption()`, `blendShallowAndDeepColors()`.

<span style="color:red">OceanAnimationLoop.js</span>
Imports: `CoreEngineInit.js`, `FluidAndWaterBase.js`, `ParticleAndWindSystem.js`
Functions & Algorithms: `updateGerstnerWaveTime()`, `animateFoamParticles()`, `renderOceanFrame()`.

---

### IMAGE 2: ARID DESERT (Dunes and Mountains)

<span style="color:red">DesertSceneConfig.js</span>
Imports: `CoreEngineInit.js`, `AtmosphericFogAndMood.js`
Functions & Algorithms: `configureDesertCamera()`, `setAridFogDensity()`, `applyOppressiveMoodColors()`.

<span style="color:red">DuneTerrainGenerator.js</span>
Imports: `BaseProceduralGeometry.js`
Functions & Algorithms: `generateRollingDunes()`, `applyFractalBrownianMotion()`, `smoothDuneNormals()`.

<span style="color:red">DesertRockFormations.js</span>
Imports: `BaseProceduralGeometry.js`, `InstancingAndColorVariation.js`
Functions & Algorithms: `generateScatteredDesertRocks()`, `applyHeatHazeDistortion()`, `configureOchreColorPalette()`.

<span style="color:red">AridSkyAndHeatHaze.js</span>
Imports: `ProceduralSkyAndClouds.js`
Functions & Algorithms: `setStarkBlueGradient()`, `configureDustScatteringHorizon()`, `applyScreenSpaceHeatHaze()`.

<span style="color:red">DesertCelShadingConfig.js</span>
Imports: `AnimeCelShadingPipeline.js`
Functions & Algorithms: `configureSandLightingQuantization()`, `setLowShadowThresholds()`, `applyRimLightingToDunes()`.

<span style="color:red">MountainPeakGenerator.js</span>
Imports: `BaseProceduralGeometry.js`
Functions & Algorithms: `generateJaggedPeaks()`, `applyFaultFormationAlgorithm()`, `calculateDistantAtmosphericScattering()`.

<span style="color:red">DesertLightingSetup.js</span>
Imports: `DynamicLightingAndShadows.js`
Functions & Algorithms: `configureLowAngleSun()`, `calculateElongatedShadows()`, `applyWarmShadowTints()`.

<span style="color:red">SandColorVariation.js</span>
Imports: `InstancingAndColorVariation.js`, `BaseProceduralGeometry.js`
Functions & Algorithms: `generateSandColorPalette()`, `applyVertexColorVariation()`, `calculateTroughDarkening()`.

<span style="color:red">DesertWindParticles.js</span>
Imports: `ParticleAndWindSystem.js`
Functions & Algorithms: `initSandParticles()`, `calculateWindDirection()`, `blendParticleOpacity()`.

<span style="color:red">DesertHorizonFog.js</span>
Imports: `AtmosphericFogAndMood.js`
Functions & Algorithms: `setupDesertFog()`, `calculateHorizonBlend()`, `applyDustColorShift()`.

---

### IMAGE 3: RIVER CANYON (Deep Gorge and Flowing Water)

<span style="color:red">CanyonSceneConfig.js</span>
Imports: `CoreEngineInit.js`, `AtmosphericFogAndMood.js`
Functions & Algorithms: `configureCanyonCamera()`, `setEnclosedFogDensity()`, `applyMajesticMoodColors()`.

<span style="color:red">CanyonWallExtruder.js</span>
Imports: `BaseProceduralGeometry.js`
Functions & Algorithms: `generateCanyonPath()`, `extrudeCanyonWalls()`, `applyVerticalNoiseStriations()`.

<span style="color:red">RiverFlowShaderConfig.js</span>
Imports: `FluidAndWaterBase.js`
Functions & Algorithms: `configureRiverVertexShader()`, `calculateFlowDirection()`, `applyEmeraldToTealGradient()`.

<span style="color:red">CanyonRockMaterialConfig.js</span>
Imports: `AnimeCelShadingPipeline.js`, `BaseProceduralGeometry.js`
Functions & Algorithms: `applyStratifiedColorLayers()`, `calculateGeometricAmbientOcclusion()`, `configureTerracottaPalette()`.

<span style="color:red">RiverReflectionPass.js</span>
Imports: `FluidAndWaterBase.js`, `DynamicLightingAndShadows.js`
Functions & Algorithms: `initRiverReflectionTarget()`, `calculatePlanarReflectionUVs()`, `blendWaterAndBedColors()`.

<span style="color:red">CanyonLightingSetup.js</span>
Imports: `DynamicLightingAndShadows.js`
Functions & Algorithms: `setupDirectionalSun()`, `setupHemisphereSkyGround()`, `calculateCanyonOcclusion()`.

<span style="color:red">RiverbedGeometryConfig.js</span>
Imports: `BaseProceduralGeometry.js`
Functions & Algorithms: `generateRiverbed()`, `smoothRiverbedDisplacement()`, `applySubmergedRockColors()`.

<span style="color:red">CanyonVegetationSpots.js</span>
Imports: `VegetationAndSSSSystem.js`, `InstancingAndColorVariation.js`
Functions & Algorithms: `placeVegetationOnLedges()`, `applyCheapSSS()`, `randomizeGreenHues()`.

<span style="color:red">CanyonAtmosphereConfig.js</span>
Imports: `AtmosphericFogAndMood.js`
Functions & Algorithms: `setupCanyonFog()`, `applyCoolBlueGreyShadowTint()`, `updateFogParameters()`.

<span style="color:red">RiverFlowAnimation.js</span>
Imports: `FluidAndWaterBase.js`, `CoreEngineInit.js`
Functions & Algorithms: `animateWaterVertices()`, `calculateFlowSpeed()`, `updateUVScrolling()`.

---

### IMAGE 4: ALPINE CLIFF (Meadows and Verticality)

<span style="color:red">CliffSceneConfig.js</span>
Imports: `CoreEngineInit.js`, `AtmosphericFogAndMood.js`
Functions & Algorithms: `configureCliffCamera()`, `setHighAltitudeFogDensity()`, `applyVibrantMoodColors()`.

<span style="color:red">CliffFaceGenerator.js</span>
Imports: `BaseProceduralGeometry.js`
Functions & Algorithms: `generateCliffBase()`, `applyHorizontalNoise()`, `createLedges()`.

<span style="color:red">AlpineVegetationSystem.js</span>
Imports: `VegetationAndSSSSystem.js`, `InstancingAndColorVariation.js`
Functions & Algorithms: `initTreeInstances()`, `initGrassInstances()`, `applyHeightBasedColorGradient()`.

<span style="color:red">AlpineSkyAndClouds.js</span>
Imports: `ProceduralSkyAndClouds.js`
Functions & Algorithms: `createAlpineSky()`, `generateVolumetricCloudNoise()`, `applyCloudBottomGreyTint()`.

<span style="color:red">CliffCelShadingConfig.js</span>
Imports: `AnimeCelShadingPipeline.js`
Functions & Algorithms: `quantizeRockLighting()`, `applyStrongRimLighting()`, `configureSoftShadowBands()`.

<span style="color:red">CliffLightingSetup.js</span>
Imports: `DynamicLightingAndShadows.js`
Functions & Algorithms: `setupSoftDiffuseLight()`, `calculateHighAltitudeScattering()`, `updateLightColors()`.

<span style="color:red">VegetationSSSConfig.js</span>
Imports: `VegetationAndSSSSystem.js`
Functions & Algorithms: `configureSSSWrapFactor()`, `calculateBacklightColorShift()`, `applyLimeToForestGradient()`.

<span style="color:red">DistantOceanBackground.js</span>
Imports: `FluidAndWaterBase.js`, `AtmosphericFogAndMood.js`
Functions & Algorithms: `createDistantOceanPlane()`, `applyHazyPaleBlueColor()`, `calculateHorizonBlend()`.

<span style="color:red">CliffWindAnimation.js</span>
Imports: `ParticleAndWindSystem.js`, `VegetationAndSSSSystem.js`
Functions & Algorithms: `animateVegetationVertices()`, `calculateWindStrength()`, `applySwayingMotion()`.

<span style="color:red">CliffAtmosphereConfig.js</span>
Imports: `AtmosphericFogAndMood.js`
Functions & Algorithms: `setupAlpineFog()`, `calculateAltitudeFading()`, `applyEtherealMoodBlend()`.

---

### IMAGE 5: TEMPERATE RURAL (Pastoral House and Gardens)

<span style="color:red">HouseSceneConfig.js</span>
Imports: `CoreEngineInit.js`, `AtmosphericFogAndMood.js`
Functions & Algorithms: `configureHouseCamera()`, `setPastoralFogDensity()`, `applyNostalgicMoodColors()`.

<span style="color:red">HouseGeometryBuilder.js</span>
Imports: `BaseProceduralGeometry.js`
Functions & Algorithms: `buildHouseWalls()`, `buildRoof()`, `buildWindowsAndDoors()`, `computePreciseNormals()`.

<span style="color:red">HouseMaterialConfig.js</span>
Imports: `AnimeCelShadingPipeline.js`
Functions & Algorithms: `applyCreamyWallColors()`, `calculateBurntRedRoofGradient()`, `configureArchitecturalShading()`.

<span style="color:red">GardenVegetationSystem.js</span>
Imports: `VegetationAndSSSSystem.js`, `InstancingAndColorVariation.js`
Functions & Algorithms: `initTreeInstances()`, `initBushInstances()`, `applyPoissonDiskSampling()`.

<span style="color:red">HouseCelShadingConfig.js</span>
Imports: `AnimeCelShadingPipeline.js`
Functions & Algorithms: `quantizeArchitecturalLighting()`, `applyPreciseShadowBoundaries()`, `configureDappledShadowNoise()`.

<span style="color:red">HouseLightingSetup.js</span>
Imports: `DynamicLightingAndShadows.js`
Functions & Algorithms: `setupLateAfternoonSun()`, `placeInteriorPointLights()`, `calculateSoftLongShadows()`.

<span style="color:red">PicturesqueSkyAndClouds.js</span>
Imports: `ProceduralSkyAndClouds.js`
Functions & Algorithms: `createPicturesqueSky()`, `generateCumulusCloudSpheres()`, `applyCloudBottomBlueTint()`.

<span style="color:red">VegetationDappleConfig.js</span>
Imports: `VegetationAndSSSSystem.js`
Functions & Algorithms: `applyCheapSSSWrap()`, `generateProceduralDappleMask()`, `randomizeMidGreenHues()`.

<span style="color:red">HousePathAndGround.js</span>
Imports: `BaseProceduralGeometry.js`
Functions & Algorithms: `generateGroundPlane()`, `applyPathGeometry()`, `calculateGroundColorVariation()`.

<span style="color:red">HouseAtmosphereConfig.js</span>
Imports: `AtmosphericFogAndMood.js`
Functions & Algorithms: `setupPastoralFog()`, `calculateDistanceFading()`, `updateFogDensity()`.

