## BATCH 16: ARCHITECTURAL EXPANSION OF BASEPROCEDURALGEOMETRY.JS SUB-MODULES

To generate the vast, stylized landscapes and complex environmental reliefs seen in the reference images while maintaining strict adherence to the anime aesthetic and high-performance requirements, the `BaseProceduralGeometry.js` module is decomposed into 10 highly specialized sub-files. These files manage vertex manipulation, normal calculation, shape carving, and context-aware terrain generation.

<span style="color:red">TerrainChunkManager.js</span>
Imports: `BufferGeometry.js`, `Vector3.js`, `Frustum.js`, `ObjectPool.js`
Functions & Algorithms: `partitionWorldIntoChunks()`, `calculateChunkLOD()`, `streamGeometryData()`, `manageChunkLifecycle()`.
Context-Aware Integration: Divides the massive procedural world (like the endless Desert or Ocean) into manageable spatial chunks. It dynamically adjusts the vertex density of each chunk based on its distance from the camera. Chunks in the foreground (like the House garden) retain maximum detail for precise cel-shading, while distant chunks are aggressively simplified to save GPU processing power.

<span style="color:red">VertexDisplacementEngine.js</span>
Imports: `MathUtils.js`, `Vector3.js`, `NoiseGenerator.js`, `BufferAttribute.js`
Functions & Algorithms: `evaluateFractalBrownianMotion()`, `applyWorleyNoiseDisplacement()`, `calculateVertexOffsetVectors()`, `updatePositionAttributes()`.
Context-Aware Integration: The core mathematical engine for shaping the terrain. It displaces vertices along their normals using layered noise functions. The noise parameters are strictly bound to the active biome: it uses low-frequency, high-amplitude noise for the rolling Arid Desert dunes, and high-frequency, jagged noise for the sharp Coastal Oceanic rocks and Alpine Cliff faces.

<span style="color:red">FlatNormalCalculator.js</span>
Imports: `Vector3.js`, `BufferAttribute.js`, `MathUtils.js`, `Face3.js`
Functions & Algorithms: `calculateFaceCentroids()`, `computePerFaceNormals()`, `discardVertexInterpolation()`, `flattenShadingBoundaries()`.
Context-Aware Integration: Absolutely critical for the anime aesthetic. Standard 3D geometry interpolates normals across vertices, creating smooth, realistic shading. This module mathematically forces all vertices of a single polygonal face to share the exact same normal vector. This creates the hard, faceted, "low-poly" look required for the `AnimeCelShadingPipeline` to render crisp, unblended shadow bands on rocks and terrain.

<span style="color:red">SignedDistanceFieldBuilder.js</span>
Imports: `Vector3.js`, `MathUtils.js`, `BufferGeometry.js`, `ShaderMaterial.js`
Functions & Algorithms: `evaluateSDFPrimitives()`, `performBooleanOperations()`, `carveCanyonPaths()`, `generateSharpReliefEdges()`.
Context-Aware Integration: Uses mathematical SDF functions to carve complex shapes directly into the geometry. It is used to cut the winding path of the River Canyon through the terrain and to create the sharp, overhanging ledges of the Alpine Cliff. The SDF evaluation ensures that the resulting geometry has perfectly clean, mathematically precise edges that render beautifully with the `OutlineRenderingController`.

<span style="color:red">BiomeShapeMorpher.js</span>
Imports: `Vector3.js`, `MathUtils.js`, `EventDispatcher.js`, `BufferAttribute.js`
Functions & Algorithms: `interpolateDisplacementAmplitudes()`, `morphNoiseFrequencies()`, `transitionTerrainProfiles()`, `smoothVertexUpdates()`.
Context-Aware Integration: Handles the seamless visual transition when the camera moves between biomes or when a climatic change alters the landscape. If the environment shifts from a dry Desert to a flooded Coastal zone, this module smoothly interpolates the vertex positions, rounding off sharp desert rocks and filling in troughs with water-level geometry in real-time.

<span style="color:red">EdgeAndSilhouetteExtractor.js</span>
Imports: `BufferGeometry.js`, `Vector3.js`, `Camera.js`, `MathUtils.js`
Functions & Algorithms: `identifySilhouetteEdges()`, `calculateEdgeCreaseAngles()`, `generateOutlineGeometryBuffers()`, `optimizeEdgeIndices()`.
Context-Aware Integration: Pre-processes the geometry to identify which edges should receive an outline. It calculates the angle between adjacent face normals; sharp angles (like the edge of a rock) are marked for thick outlines, while smooth curves are ignored. This data is passed to the `OutlineStyleManager` to ensure line art is only drawn where it enhances the anime composition.

<span style="color:red">ProceduralRockGenerator.js</span>
Imports: `IcosahedronGeometry.js`, `Vector3.js`, `MathUtils.js`, `Color.js`
Functions & Algorithms: `subdivideBasePolyhedron()`, `applyRandomVertexJitter()`, `calculateRockFacetGroups()`, `instanceRockVariations()`.
Context-Aware Integration: Specifically algorithms for generating the stylized rocks seen in the Ocean and Desert images. It starts with a base Icosahedron, randomly jitters the vertices to break symmetry, and then groups the resulting faces into distinct "facets". These facets are then passed to the `InstancingAndColorVariation` system to be colored with distinct, flat-shaded bands.

<span style="color:red">CliffAndCanyonExtruder.js</span>
Imports: `Shape.js`, `ExtrudeGeometry.js`, `Vector3.js`, `CatmullRomCurve3.js`
Functions & Algorithms: `traceCanyonSpline()`, `extrudeVerticalWalls()`, `applyStriationNoise()`, `generateOverhangGeometry()`.
Context-Aware Integration: Generates the massive vertical structures. It takes a 2D spline path and extrudes it vertically, applying horizontal noise layers to create the sedimentary striations seen in the River Canyon. It also mathematically calculates overhangs and caves, ensuring the geometry correctly receives the deep, ambient-occluded shadows calculated by the `DynamicLightingAndShadows` system.

<span style="color:red">GeometryOptimizationPipeline.js</span>
Imports: `BufferGeometry.js`, `Uint16Array.js`, `WebGLRenderer.js`, `MathUtils.js`
Functions & Algorithms: `mergeIdenticalVertices()`, `optimizeIndexBufferLayout()`, `stripUnusedAttributes()`, `calculateGeometryBoundingBoxes()`.
Context-Aware Integration: Ensures the generated geometry is perfectly optimized for the GPU. It removes duplicate vertices, reorders the index buffer for better cache locality (improving vertex shader throughput), and strips out unnecessary attributes (like UVs) if the material is purely procedural. This maximizes FPS, especially when rendering thousands of instanced rocks.

<span style="color:red">DynamicTessellationController.js</span>
Imports: `BufferGeometry.js`, `Camera.js`, `Vector3.js`, `WebGLRenderer.js`
Functions & Algorithms: `calculateScreenSpaceErrorMetric()`, `subdivideMeshFaces()`, `collapseDistantTriangles()`, `updateLODAttributes()`.
Context-Aware Integration: A real-time Level of Detail (LOD) system that operates directly on the geometry buffers. It measures how many pixels a triangle occupies on the screen. If a distant mountain face occupies less than 2 pixels, it collapses the geometry to save GPU fill rate. As the camera approaches the House or the foreground rocks, it dynamically subdivides the mesh to provide the high geometric detail needed for precise cel-shading and outlining.

