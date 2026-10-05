API Documentation — src/core/032_rnd_NoImageTexturePolicy.js

File Purpose

This file is the policy enforcement module that guarantees the anime lighting stack never depends on image textures. The engine targets a fully procedural visual identity: colors come from palette tables, gradients come from shader math, noise comes from GLSL functions, normals come from geometry plus quantization, occlusion comes from algorithms, and the sky comes from scattering math. No PNG, no JPG, no WebP, no KTX, no DDS, no HDR, no EXR, no Basis, no ASTC — no bitmap ever.

The policy exists to solve five concrete problems on Android mobile:

Zero image decode on the main thread. Decoding a 2048×2048 PNG costs 30–80 ms on a mid-range Android device, and that cost lands on the main thread. On a device that just finished booting, that is a guaranteed dropped frame. A fully procedural pipeline never decodes anything.

Zero GPU memory spent on bitmaps. An uncompressed RGBA8 2048×2048 texture is 16 MB. On a 4 GB Android device, the GPU memory budget is often shared with the CPU, and a handful of texture allocations can push the renderer into OOM territory. The procedural pipeline spends its GPU memory on render targets and geometry, not on bitmaps.

Zero network requests. No CDN stalls, no cache misses, no "texture not loaded yet" race conditions. The engine is ready to render the moment its shaders compile.

Deterministic boot. The procedural engine does not have an async texture-loading phase. There is no "wait for 40 textures to finish loading" screen. The engine either runs or it fails at shader compile time.

Deterministic look. The anime style is derived from numbers, so it is resolution-independent and device-independent. A shader that renders at 4K looks the same as a shader that renders at 720p, just sharper.

The module enforces the policy in two ways:

1. Runtime auditing. A NoImageTexturePolicy instance tracks every texture, material, and loader it encounters and records a violation for any that derive from a bitmap source.
2. Sanctioned alternatives. The module exports four procedural texture factories — createProceduralLUT, createProceduralNoise, createProceduralFloatTexture, createProceduralDataArrayTexture — that any subsystem can use to create GPU textures from pure math. These are the only sanctioned way to create a DataTexture.

The policy has four enforcement modes:

· AUDIT — passive scan, records violations, no throw, no log.
· WARN — scan plus log warning per violation.
· ENFORCE — scan plus throw on first violation.
· PARANOID — scan at boot and again every N frames.

Every audit runs OFF the per-frame hot path. Boot audits run once. Paranoid audits run on a throttled cadence. The render loop never touches the policy.

---

Exported Constants

PERF_TIER_LOCAL

Internal. The cached PERF_TIER string from getPerfTier().

POLICY_MODE

Type: frozen enum

The four enforcement modes.

· AUDIT = 0 — passive scan.
· WARN = 1 — scan plus warning.
· ENFORCE = 2 — scan plus throw.
· PARANOID = 3 — scan at boot and on a schedule.

POLICY_MODE_NAME

Type: frozen array

Values: ['audit', 'warn', 'enforce', 'paranoid'].

VIOLATION_KIND

Type: frozen enum

The violation categories.

· UNKNOWN = 0 — a texture source could not be classified.
· LOADER_PRESENT = 1 — a forbidden loader exists on the THREE namespace.
· HTML_IMAGE_TEXTURE = 2 — a texture whose image is an HTMLImageElement.
· CANVAS_TEXTURE = 3 — a texture whose image is an HTMLCanvasElement or OffscreenCanvas.
· IMAGE_BITMAP = 4 — a texture whose image is an ImageBitmap.
· VIDEO_TEXTURE = 5 — a texture whose image is an HTMLVideoElement.
· COMPRESSED = 6 — a CompressedTexture.
· CUBE_FROM_IMAGES = 7 — a CubeTexture built from six image faces.
· MATERIAL_MAP = 8 — a material map slot points at a bitmap-backed texture.
· ENV_MAP = 9 — a material envMap points at a bitmap-backed texture.
· DATA_TEXTURE_UNMARKED = 10 — a DataTexture that has not been marked procedural.
· PMREM_UNMARKED = 11 — a PMREM output that has not been marked procedural.

VIOLATION_KIND_NAME

Type: frozen array

Maps the enum to strings: ['unknown', 'loader_present', 'html_image_texture', 'canvas_texture', 'image_bitmap', 'video_texture', 'compressed', 'cube_from_images', 'material_map', 'env_map', 'data_texture_unmarked', 'pmrem_unmarked'].

MAX_VIOLATIONS

Type: number

Value: 64 on HIGH, 48 on MEDIUM, 32 on LOW.

The maximum number of violation records retained in the ring buffer.

FORBIDDEN_MATERIAL_MAP_SLOTS

Type: frozen array

The list of material map slots that must never point at a bitmap texture. Includes map, normalMap, bumpMap, roughnessMap, metalnessMap, emissiveMap, aoMap, alphaMap, lightMap, envMap, displacementMap, specularMap, clearcoatMap, clearcoatNormalMap, clearcoatRoughnessMap, iridescenceMap, iridescenceThicknessMap, sheenColorMap, sheenRoughnessMap, transmissionMap, thicknessMap, and anisotropyMap.

FORBIDDEN_LOADER_NAMES

Type: frozen array

The list of loader class names that, if present on the THREE namespace, imply an attempt to load bitmaps. Includes TextureLoader, ImageLoader, CubeTextureLoader, RGBELoader, EXRLoader, HDRLoader, KTXLoader, KTX2Loader, DDSLoader, PVRLoader, BasisTextureLoader, CompressedTextureLoader, TGALoader, PDBLoader, MTLLoader, FBXLoader, GLTFLoader, and OBJLoader.

PROCEDURAL_FLAG and PROCEDURAL_SOURCE

Internal. String constants '__procedural' and '__proceduralSource' used as userData keys on marked textures.

---

Module-Level State (Not Exported Directly)

_defaultPolicy

Type: NoImageTexturePolicy | null

The module-level singleton.

---

Internal Helper Functions (Documented)

_now()

Returns: the current high-resolution timestamp.

_isTexture(obj)

Parameters: obj — any value.

Returns: true if the object is a Texture, a CompressedTexture, or a CubeTexture.

_isDataTexture(obj)

Returns: true if the object is a DataTexture.

_isDataArrayTexture(obj)

Returns: true if the object is a DataArrayTexture.

_isCubeTexture(obj)

Returns: true if the object is a CubeTexture.

_isCompressedTexture(obj)

Returns: true if the object is a CompressedTexture.

_isCanvasTexture(obj)

Returns: true if the object is a CanvasTexture.

_isRenderTargetTexture(obj)

Returns: true if the object is a RenderTargetTexture. These are always allowed because they are GPU-generated, not bitmap-sourced.

_isDepthTexture(obj)

Returns: true if the object is a DepthTexture. Always allowed.

_isPMREM(obj)

Returns: true if the object is a PMREMGenerator.

_isProceduralFlagged(obj)

Parameters: obj — any value.

Returns: true if the object's userData.__procedural is set to true.

Purpose: the ONLY sanctioned opt-out. A texture is allowed to exist without being a DataTexture if it has been marked procedural via markProcedural.

---

Exported Class — ViolationRecord

One instance per recorded violation.

Constructor

```
new ViolationRecord()
```

Instance Properties

· kind — one of VIOLATION_KIND.
· label — a diagnostic label.
· object — the offending object.
· details — a string describing the specific issue.
· frame — the frame at which the violation was recorded.
· timeMs — the timestamp.

Instance Methods

reset()

Returns: this. Zeroes every field.

---

Exported Class — NoImageTexturePolicy

The main policy enforcer.

Constructor

```
new NoImageTexturePolicy(options = {})
```

Parameters:

· mode — one of POLICY_MODE. Default WARN.
· channel — the logger channel. Default LOG_CHANNEL.CORE.
· logViolations — if true, logs each violation. Default true.
· recordProfilerMark — if true, records a profiler marker per violation. Default true.
· throwOnViolation — if true, a violation throws. Default false.
· maxViolations — the ring buffer size. Default MAX_VIOLATIONS.
· paranoidIntervalFrames — the frame interval for PARANOID scans. Default 600.
· allowCompressed — if true, CompressedTexture is allowed. Default false.
· allowCanvasTexture — if true, marked CanvasTexture is allowed. Default false.
· allowPMREM — if true, marked PMREM outputs are allowed. Default true.

Constructor work:

1. Stores options and the mode.
2. Allocates the violation ring.
3. Initializes the counters: auditCount, auditSceneCount, auditMaterialCount, auditTextureCount.
4. Initializes frame and lastParanoidFrame.
5. Allocates _listeners (Map).
6. If throwOnViolation, creates a boundary for violation enforcement.

Instance Properties

· mode — the current policy mode.
· violations — the violation ring array.
· violationHead — the ring write pointer.
· violationCount — the number of valid entries.
· violationTotal — the total violations ever recorded.
· auditCount — the number of enforceNoImageTextures calls.
· auditSceneCount — the number of scene audits.
· auditMaterialCount — the number of material audits.
· auditTextureCount — the number of texture audits.
· frame — the current frame.

Instance Methods

on(event, fn)

Parameters:

· event — one of 'violation', 'audit-complete', 'paranoid-tick'.
· fn — the callback.

Returns: unsubscribe function.

off(event, fn)

Returns: nothing.

_emit(event, payload)

Internal. Dispatches to listeners.

beginFrame(frameNumber)

Parameters: frameNumber — the current frame number.

Returns: nothing.

Purpose: sets the frame and, if the mode is PARANOID, checks whether the throttle interval has elapsed since the last paranoid scan. When it has, emits paranoid-tick so an external system can trigger a re-audit.

_record(kind, label, object, details)

Internal. Records a violation. Writes to the ring, updates the counters, logs (if enabled), records a profiler marker (if enabled), emits violation, and throws if the mode is ENFORCE and throwOnViolation is set.

markProcedural(texture, source)

Parameters:

· texture — the texture to mark.
· source — an optional string describing where it came from.

Returns: boolean.

Purpose: the ONLY sanctioned way to opt a texture out of the policy. Sets userData.__procedural = true and optionally userData.__proceduralSource = source. Used by the procedural texture factories below.

isProcedural(texture)

Parameters: texture — the texture to query.

Returns: boolean.

unmarkProcedural(texture)

Parameters: texture — the texture to unmark.

Returns: boolean.

auditTexture(texture, label)

Parameters:

· texture — the texture to audit.
· label — a diagnostic label.

Returns: boolean — true if compliant, false if a violation was recorded.

Flow:

1. If the argument is not a texture, returns true (nothing to check).
2. Increments auditTextureCount.
3. Allow-list: render target textures and depth textures are always allowed.
4. Allow-list: marked procedural textures are allowed.
5. CompressedTexture is always a violation unless allowCompressed is set.
6. CubeTexture is audited via _auditCubeTexture.
7. CanvasTexture is allowed only if allowCanvasTexture and the texture is procedural-flagged.
8. DataTexture is allowed only if procedural-flagged.
9. Regular Texture: checks the .image source.
   · HTMLImageElement → violation.
   · ImageBitmap → violation.
   · HTMLVideoElement → violation.
   · HTMLCanvasElement → allowed only if procedural-flagged.
   · OffscreenCanvas → allowed only if procedural-flagged.
   · Generic object with width and height → allowed only if procedural-flagged.

_auditCubeTexture(cube, label)

Internal. Iterates the six faces of a CubeTexture and audits each.

auditMaterial(material, label)

Parameters:

· material — the material to audit.
· label — a diagnostic label.

Returns: boolean.

Flow:

1. Increments auditMaterialCount.
2. Iterates FORBIDDEN_MATERIAL_MAP_SLOTS.
3. For each slot with a non-null value, calls auditTexture.
4. If the slot is envMap, records VIOLATION_KIND.ENV_MAP on failure. Otherwise records VIOLATION_KIND.MATERIAL_MAP.

auditScene(scene, label)

Parameters:

· scene — the scene to traverse.
· label — a diagnostic label.

Returns: the number of violations found in this pass.

Flow:

1. Increments auditSceneCount.
2. Traverses the scene with scene.traverse.
3. For every object with a material, calls auditMaterial.
4. Audits scene.background if it is a texture.
5. Audits scene.environment if it is a texture.
6. Returns the difference between the current total and the total at the start of the pass.

auditThreeNamespace()

Returns: the number of forbidden loaders found.

Purpose: iterates FORBIDDEN_LOADER_NAMES and checks whether each exists on the THREE namespace. If any does, records VIOLATION_KIND.LOADER_PRESENT.

enforceNoImageTextures(scene)

Parameters: scene — an optional scene to audit.

Returns: an object with loaders, sceneViolations, totalViolations, durationMs, and ok.

Purpose: the boot hook. Runs the namespace audit, then the scene audit if a scene was supplied. Emits audit-complete.

isCompliant()

Returns: true if no violations have been recorded.

getRecentViolations(max, out)

Parameters:

· max — the maximum number of violations to copy.
· out — an array to receive them.

Returns: the number of records copied.

getStats()

Returns: an object with mode, modeIndex, compliant, violationTotal, violationRingCount, violationRingCap, auditCount, auditSceneCount, auditMaterialCount, auditTextureCount, allowCompressed, allowCanvasTexture, allowPMREM.

reset()

Returns: this. Clears the violation ring and zeroes every counter.

dispose()

Returns: nothing. Resets and clears the listener map.

---

Exported Convenience Functions

These delegate to the module-level singleton.

markProcedural(texture, source)

Marks a texture as procedural.

isProcedural(texture)

Queries the procedural flag.

enforceNoImageTextures(scene)

Runs the boot audit.

auditNoImageScene(scene, label)

Runs a scene audit.

auditNoImageMaterial(material, label)

Runs a material audit.

auditNoImageTexture(texture, label)

Runs a texture audit.

noImagePolicyBeginFrame(frameNumber)

Sets the frame on the default policy.

isNoImagePolicyCompliant()

Returns the compliance status.

---

Exported Procedural Texture Factories

These are the only sanctioned way to create GPU textures in the engine.

createProceduralLUT(size, fillFn)

Parameters:

· size — the number of samples. Default 256.
· fillFn — a function (t, r, g, b) => void that fills the RGB values at normalized time t ∈ [0, 1]. The r, g, b arguments are single-element Float32Array buffers used as output.

Returns: a THREE.DataTexture with the linear filter, ClampToEdge wrapping, no mipmaps, and the __procedural flag set.

Purpose: creates a 1D lookup table for palette or gradient sampling. The texture occupies size × 4 bytes.

createProceduralNoise(size, fillFn)

Parameters:

· size — the texture dimensions. Default 128.
· fillFn — a function (x, y, r, g, b) => void that fills the RGB values at pixel (x, y).

Returns: a THREE.DataTexture with the linear filter, RepeatWrapping, no mipmaps, and the __procedural flag.

Purpose: creates a 2D noise texture from pure math.

createProceduralFloatTexture(width, height, fillFn)

Parameters:

· width — the texture width.
· height — the texture height.
· fillFn — a function (x, y, w, h, r, g, b, a) => void that fills the RGBA values.

Returns: a THREE.DataTexture with RGBAFormat, FloatType, linear filter, ClampToEdge, no mipmaps, and the __procedural flag.

Purpose: creates a high-precision float texture. Used for SH coefficients, GI probe grids, and linear-space palette LUTs.

createProceduralDataArrayTexture(width, height, depth, fillFn)

Parameters:

· width, height, depth — the dimensions.
· fillFn — a function (x, y, z, r, g, b, a) => void that fills the RGBA values.

Returns: a THREE.DataArrayTexture with RGBAFormat, UnsignedByteType, linear filter, ClampToEdge, no mipmaps, and the __procedural flag.

Purpose: creates a 3D texture for color grading LUTs, GI SH lattices, or any 3D lookup table.

---

Exported Functions

getDefaultNoImagePolicy()

Returns: the module-level singleton NoImageTexturePolicy, creating it on first call.

disposeDefaultNoImagePolicy()

Returns: nothing.

createNoImagePolicy(options = {})

Returns: a new NoImageTexturePolicy.

---

Default Export

The default export bundles: NoImageTexturePolicy, ViolationRecord, createNoImagePolicy, getDefaultNoImagePolicy, disposeDefaultNoImagePolicy, markProcedural, isProcedural, enforceNoImageTextures, auditNoImageScene, auditNoImageMaterial, auditNoImageTexture, noImagePolicyBeginFrame, isNoImagePolicyCompliant, createProceduralLUT, createProceduralNoise, createProceduralFloatTexture, createProceduralDataArrayTexture, POLICY_MODE, POLICY_MODE_NAME, VIOLATION_KIND, VIOLATION_KIND_NAME, FORBIDDEN_MATERIAL_MAP_SLOTS, FORBIDDEN_LOADER_NAMES, MAX_VIOLATIONS.

---

Usage Pattern

The bootstrap runs the boot audit once:

```
import {
  enforceNoImageTextures,
  createNoImagePolicy,
  POLICY_MODE,
} from './src/core/032_rnd_NoImageTexturePolicy.js';

const result = enforceNoImageTextures(scene);
if (!result.ok) {
  console.warn(`No-image policy violations: ${result.totalViolations}`);
}
```

A subsystem that needs a procedural LUT:

```
import { createProceduralLUT } from './src/core/032_rnd_NoImageTexturePolicy.js';

const paletteLUT = createProceduralLUT(256, (t, r, g, b) => {
  const rgb = samplePaletteAt(t);
  r[0] = rgb[0];
  g[0] = rgb[1];
  b[0] = rgb[2];
});

material.uniforms.uPaletteLUT.value = paletteLUT;
```

A subsystem that creates a procedural noise texture:

```
import { createProceduralNoise } from './src/core/032_rnd_NoImageTexturePolicy.js';

const noiseTex = createProceduralNoise(128, (x, y, r, g, b) => {
  const fx = x / 128;
  const fy = y / 128;
  const n = fbm2D(fx * 8, fy * 8, 4);
  r[0] = n;
  g[0] = n;
  b[0] = n;
});

material.uniforms.uNoiseTex.value = noiseTex;
```

A debug HUD that monitors compliance:

```
import { getDefaultNoImagePolicy } from './src/core/032_rnd_NoImageTexturePolicy.js';

const policy = getDefaultNoImagePolicy();
if (!policy.isCompliant()) {
  const violations = [];
  const n = policy.getRecentViolations(5, violations);
  for (let i = 0; i < n; i++) {
    console.warn(`Violation: ${violations[i].kind} — ${violations[i].label}`);
  }
}
```

A paranoid enforcement build that re-audits once every 600 frames:

```
const policy = createNoImagePolicy({
  mode: POLICY_MODE.PARANOID,
  paranoidIntervalFrames: 600,
});

policy.on('paranoid-tick', () => {
  policy.auditScene(scene, 'periodic');
});
```

The policy is what makes the anime lighting stack's "no images, ever" guarantee real. Without it, a developer could accidentally add a TextureLoader call and pull a bitmap into the engine, and nothing would catch it until the visual output stopped matching the reference images. With it, the violation is caught at boot, logged with a specific category, and — in ENFORCE mode — thrown before the shader ever sees the texture.

The four procedural factories make the sanctioned path easy. Any subsystem that needs a lookup table, a noise texture, a high-precision float texture, or a 3D LUT can create one from pure math and mark it procedural in one call. The policy audits do not need to distinguish these from any other texture because the __procedural flag on the userData is the single source of truth.

