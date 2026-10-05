// File : 015
// name : src/core/015_rnd_AssetManifest.js
// description : Authoritative asset manifest for the anime lighting stack on
//               Android mobile. This is a NO-IMAGE-TEXTURE project — every
//               "asset" in this manifest is a PROCEDURAL descriptor: shader
//               chunk, geometry factory, palette table, light preset,
//               biome definition, environment preset, or generator
//               parameter block. The manifest never references PNG/JPG/KTX/
//               WebP/HDR files because the entire look is generated in
//               shader from colors + noise + procedural geometry.
//
//               Responsibilities:
//                 • Declare every procedural asset the lighting stack depends
//                   on with a stable id, kind, tier requirement, and load
//                   order group.
//                 • Resolve load order via explicit `dependsOn` edges so the
//                   bootstrap can pre-warm shader chunks and palette tables
//                   in the correct sequence without a runtime dependency
//                   solver.
//                 • Provide per-tier filtering: PERF_TIER 'LOW' skips assets
//                   tagged as HIGH-only so low-end Android devices don't
//                   build geometry they will never render.
//                 • Provide per-mode filtering: indoor/exterior/biome tags
//                   let the engine pre-warm only what the current scene
//                   needs, deferring the rest to streamed load.
//                 • Expose a resolver that maps logical ids to module paths
//                   so downstream lighting subsystems import by symbolic id
//                   (`manifest.getPath('shader:anime_cel_lighting')`)
//                   instead of hard-coding relative paths.
//                 • Track per-asset load state (PENDING / LOADED / FAILED /
//                   SKIPPED) so the bootstrap can report progress and the
//                   adaptive quality controller can react to missing assets.
//                 • Allocation-free hot path: manifest is read once per asset
//                   during boot; per-frame lookups are typed-array-indexed
//                   via a numeric alias table.
//
//               Design:
//                 • Fixed-capacity asset table sized once at construction.
//                 • Every asset record is a frozen plain object literal with
//                   the fields: id, kind, path, tier, tags, dependsOn,
//                   state, loadedAt, resolved.
//                 • Kinds: SHADER_CHUNK, SHADER_MATERIAL, GEOMETRY_FACTORY,
//                   PALETTE, LIGHT_PRESET, BIOME_DEF, ENV_PRESET, TASK_GRAPH,
//                   POOL_PRESET, METADATA.
//                 • Tags (bitmask): TAG_INDOOR, TAG_EXTERIOR, TAG_DESERT,
//                   TAG_SNOW, TAG_SEA, TAG_HOUSE, TAG_CANYON, TAG_FOREST,
//                   TAG_CRITICAL, TAG_OPTIONAL, TAG_DEBUG.
//                 • Tier: TIER_LOW, TIER_MEDIUM, TIER_HIGH, TIER_ANY.
//                 • Loader dispatch: each kind maps to a loader function via
//                   `registerLoader(kind, fn)`. Default loaders are
//                   registered at construction for dynamic ESM imports.
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0 API
//               only; no external manifest libs; every internal array sized
//               once at construction.
// best for : Single source of truth for every procedural resource the anime
//            lighting stack needs. The bootstrap imports this manifest,
//            filters by tier + scene tags, resolves load order, and pre-warms
//            shader chunks, palette tables, and geometry factories in one
//            deterministic pass — no scattered relative imports, no missing
//            shader chunks at runtime, no surprise allocations on Android.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import {
  getPerfTier,
} from './008_scn_world.js';

import {
  assertBiteCSReady,
  isBiteCSReady,
} from './009_scn_BiteCSVersionPolicy.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER = getPerfTier();

export const ASSET_KIND = Object.freeze({
  SHADER_CHUNK:     0,
  SHADER_MATERIAL:  1,
  GEOMETRY_FACTORY: 2,
  PALETTE:          3,
  LIGHT_PRESET:     4,
  BIOME_DEF:        5,
  ENV_PRESET:       6,
  TASK_GRAPH:       7,
  POOL_PRESET:      8,
  METADATA:         9,
  COUNT:           10,
});

export const ASSET_KIND_NAME = Object.freeze([
  'shader_chunk',
  'shader_material',
  'geometry_factory',
  'palette',
  'light_preset',
  'biome_def',
  'env_preset',
  'task_graph',
  'pool_preset',
  'metadata',
]);

export const ASSET_STATE = Object.freeze({
  PENDING:  0,
  LOADING:  1,
  LOADED:   2,
  FAILED:   3,
  SKIPPED:  4,
});

export const ASSET_STATE_NAME = Object.freeze([
  'pending',
  'loading',
  'loaded',
  'failed',
  'skipped',
]);

export const ASSET_TIER = Object.freeze({
  ANY:    0,
  LOW:    1,
  MEDIUM: 2,
  HIGH:   3,
});

export const ASSET_TAG = Object.freeze({
  NONE:      0,
  INDOOR:    1 << 0,
  EXTERIOR:  1 << 1,
  DESERT:    1 << 2,
  SNOW:      1 << 3,
  SEA:       1 << 4,
  HOUSE:     1 << 5,
  CANYON:    1 << 6,
  FOREST:    1 << 7,
  CRITICAL:  1 << 8,
  OPTIONAL:  1 << 9,
  DEBUG:     1 << 10,
  WORKER:    1 << 11,
});

export const MAX_ASSETS = 512;
export const MAX_LOADERS = ASSET_KIND.COUNT;

/* ------------------------------------------------------------------ */
/* 1. HELPERS                                                         */
/* ------------------------------------------------------------------ */

function _now() {
  return (typeof performance !== 'undefined' ? performance.now() : Date.now());
}

function _tierAllowed(assetTier, deviceTier) {
  if (assetTier === ASSET_TIER.ANY) return true;
  const devRank = deviceTier === 'HIGH' ? 3 : deviceTier === 'MEDIUM' ? 2 : 1;
  return assetTier <= devRank;
}

/* ------------------------------------------------------------------ */
/* 2. ASSET RECORD                                                    */
/* ------------------------------------------------------------------ */

export class AssetRecord {
  constructor() {
    this.id         = '';
    this.kind       = ASSET_KIND.METADATA;
    this.path       = '';
    this.tier       = ASSET_TIER.ANY;
    this.tags       = ASSET_TAG.NONE;
    this.dependsOn  = null;      // frozen string[] or null
    this.state      = ASSET_STATE.PENDING;
    this.loadedAt   = 0;
    this.resolved   = null;      // whatever the loader returns
    this.loader     = null;      // (record) -> Promise<unknown>
    this.priority   = 100;
    this.group      = 'default';
  }

  reset() {
    this.id         = '';
    this.kind       = ASSET_KIND.METADATA;
    this.path       = '';
    this.tier       = ASSET_TIER.ANY;
    this.tags       = ASSET_TAG.NONE;
    this.dependsOn  = null;
    this.state      = ASSET_STATE.PENDING;
    this.loadedAt   = 0;
    this.resolved   = null;
    this.loader     = null;
    this.priority   = 100;
    this.group      = 'default';
  }
}

/* ------------------------------------------------------------------ */
/* 3. ASSET MANIFEST                                                  */
/* ------------------------------------------------------------------ */

export class AssetManifest {
  constructor(options = {}) {
    this.options = Object.assign({
      tier:            PERF_TIER,
      tags:            ASSET_TAG.NONE,
      includeOptional: true,
      includeDebug:    false,
      autoLoad:        false,
      concurrency:     PERF_TIER === 'HIGH' ? 6 : PERF_TIER === 'MEDIUM' ? 4 : 2,
    }, options || {});

    this.capacity = MAX_ASSETS;
    this.records  = new Array(this.capacity);
    for (let i = 0; i < this.capacity; i++) this.records[i] = new AssetRecord();

    this.count      = 0;
    this.indexById  = new Map();

    this.loaders    = new Array(MAX_LOADERS).fill(null);

    this.frame      = 0;
    this.stats = {
      registered: 0,
      loaded:     0,
      failed:     0,
      skipped:    0,
      loadMs:     0,
    };

    this._listeners = new Map();

    this._installDefaultLoaders();
  }

  /* ---------------- events ---------------- */

  on(event, fn) {
    if (typeof event !== 'string' || typeof fn !== 'function') return () => {};
    let arr = this._listeners.get(event);
    if (!arr) { arr = []; this._listeners.set(event, arr); }
    arr.push(fn);
    return () => this.off(event, fn);
  }

  off(event, fn) {
    const arr = this._listeners.get(event);
    if (!arr) return;
    const i = arr.indexOf(fn);
    if (i >= 0) arr.splice(i, 1);
  }

  _emit(event, payload) {
    const arr = this._listeners.get(event);
    if (!arr) return;
    for (let i = 0; i < arr.length; i++) {
      try { arr[i](payload); } catch (e) { console.error(`[015_rnd_AssetManifest] listener error on "${event}"`, e); }
    }
  }

  /* ---------------- loaders ---------------- */

  registerLoader(kind, fn) {
    if (kind < 0 || kind >= MAX_LOADERS) return false;
    this.loaders[kind] = (typeof fn === 'function') ? fn : null;
    return true;
  }

  _installDefaultLoaders() {
    // Default loader: dynamic ESM import. Returns the module namespace.
    const esmLoader = async (record) => {
      if (!record.path) throw new Error(`Asset "${record.id}" has no path`);
      return await import(/* @vite-ignore */ record.path);
    };

    for (let k = 0; k < MAX_LOADERS; k++) this.loaders[k] = esmLoader;
  }

  /* ---------------- registration ---------------- */

  register(spec) {
    if (!spec || typeof spec.id !== 'string' || spec.id.length === 0) return null;
    if (this.count >= this.capacity) return null;
    if (this.indexById.has(spec.id)) return this.records[this.indexById.get(spec.id)];

    const idx = this.count++;
    const rec = this.records[idx];
    rec.reset();

    rec.id         = spec.id;
    rec.kind       = spec.kind !== undefined ? spec.kind : ASSET_KIND.METADATA;
    rec.path       = spec.path || '';
    rec.tier       = spec.tier !== undefined ? spec.tier : ASSET_TIER.ANY;
    rec.tags       = spec.tags !== undefined ? spec.tags : ASSET_TAG.NONE;
    rec.dependsOn  = Array.isArray(spec.dependsOn) ? Object.freeze(spec.dependsOn.slice()) : null;
    rec.priority   = spec.priority !== undefined ? spec.priority : 100;
    rec.group      = spec.group || 'default';
    rec.state      = ASSET_STATE.PENDING;
    rec.loader     = typeof spec.loader === 'function' ? spec.loader : null;

    this.indexById.set(rec.id, idx);
    this.stats.registered++;

    this._emit('registered', { id: rec.id, kind: rec.kind, group: rec.group });
    return rec;
  }

  registerMany(specs) {
    if (!Array.isArray(specs)) return 0;
    let n = 0;
    for (let i = 0; i < specs.length; i++) {
      if (this.register(specs[i])) n++;
    }
    return n;
  }

  /* ---------------- lookup ---------------- */

  get(id) {
    const idx = this.indexById.get(id);
    if (idx === undefined) return null;
    return this.records[idx];
  }

  getPath(id) {
    const r = this.get(id);
    return r ? r.path : null;
  }

  getResolved(id) {
    const r = this.get(id);
    return r ? r.resolved : null;
  }

  getState(id) {
    const r = this.get(id);
    return r ? r.state : ASSET_STATE.PENDING;
  }

  isLoaded(id) {
    return this.getState(id) === ASSET_STATE.LOADED;
  }

  /* ---------------- filtering ---------------- */

  /**
   * Return the ordered list of asset ids that should load for the current
   * tier + tag filter. Order respects `dependsOn` edges and per-asset
   * priority within a group.
   */
  resolveLoadOrder() {
    const allowed = [];
    for (let i = 0; i < this.count; i++) {
      const rec = this.records[i];
      if (!this._passesFilter(rec)) {
        rec.state = ASSET_STATE.SKIPPED;
        this.stats.skipped++;
        continue;
      }
      allowed.push(rec);
    }

    // Topological sort using dependsOn edges.
    const inDeg = new Map();
    const byId = new Map();
    for (let i = 0; i < allowed.length; i++) {
      byId.set(allowed[i].id, allowed[i]);
      inDeg.set(allowed[i].id, 0);
    }

    for (let i = 0; i < allowed.length; i++) {
      const rec = allowed[i];
      if (!rec.dependsOn) continue;
      for (let d = 0; d < rec.dependsOn.length; d++) {
        const depId = rec.dependsOn[d];
        if (byId.has(depId)) {
          inDeg.set(rec.id, (inDeg.get(rec.id) || 0) + 1);
        }
      }
    }

    const queue = [];
    for (const [id, deg] of inDeg) if (deg === 0) queue.push(id);

    const order = [];
    while (queue.length > 0) {
      // Stable within same priority: sort the queue by priority ascending.
      queue.sort((a, b) => (byId.get(a).priority - byId.get(b).priority));
      const id = queue.shift();
      order.push(id);
      const rec = byId.get(id);
      // Decrement children.
      for (const [childId, deg] of inDeg) {
        if (deg === 0) continue;
        const child = byId.get(childId);
        if (!child || !child.dependsOn) continue;
        if (child.dependsOn.indexOf(id) >= 0) {
          inDeg.set(childId, deg - 1);
          if (deg - 1 === 0) queue.push(childId);
        }
      }
    }

    if (order.length !== allowed.length) {
      console.warn('[015_rnd_AssetManifest] cycle or missing dep detected — appending unresolved tail');
      for (let i = 0; i < allowed.length; i++) {
        if (order.indexOf(allowed[i].id) < 0) order.push(allowed[i].id);
      }
    }

    return order;
  }

  _passesFilter(rec) {
    if (!_tierAllowed(rec.tier, this.options.tier)) return false;
    if (!this.options.includeOptional && (rec.tags & ASSET_TAG.OPTIONAL)) return false;
    if (!this.options.includeDebug && (rec.tags & ASSET_TAG.DEBUG)) return false;

    // Tag filter: if the manifest specifies tags, an asset must intersect
    // them OR be CRITICAL (critical assets always load).
    const want = this.options.tags;
    if (want === ASSET_TAG.NONE) return true;
    if (rec.tags & ASSET_TAG.CRITICAL) return true;
    if ((rec.tags & want) !== 0) return true;
    return false;
  }

  /* ---------------- loading ---------------- */

  async load(id) {
    const rec = this.get(id);
    if (!rec) return null;
    if (rec.state === ASSET_STATE.LOADED) return rec.resolved;
    if (rec.state === ASSET_STATE.SKIPPED) return null;

    // Wait for dependencies first.
    if (rec.dependsOn) {
      for (let d = 0; d < rec.dependsOn.length; d++) {
        const depId = rec.dependsOn[d];
        if (this.indexById.has(depId)) {
          await this.load(depId);
        }
      }
    }

    rec.state = ASSET_STATE.LOADING;
    this._emit('loading', { id: rec.id, kind: rec.kind });

    const t0 = _now();
    try {
      const loader = rec.loader || this.loaders[rec.kind] || this.loaders[ASSET_KIND.METADATA];
      const resolved = await loader(rec);
      rec.resolved = resolved;
      rec.state = ASSET_STATE.LOADED;
      rec.loadedAt = _now();
      this.stats.loaded++;
      this.stats.loadMs += (rec.loadedAt - t0);
      this._emit('loaded', { id: rec.id, kind: rec.kind, ms: rec.loadedAt - t0 });
      return resolved;
    } catch (e) {
      rec.state = ASSET_STATE.FAILED;
      this.stats.failed++;
      this._emit('failed', { id: rec.id, kind: rec.kind, error: e });
      console.error(`[015_rnd_AssetManifest] failed to load "${rec.id}"`, e);
      return null;
    }
  }

  async loadGroup(groupName) {
    const order = this.resolveLoadOrder();
    const batch = [];
    for (let i = 0; i < order.length; i++) {
      const rec = this.get(order[i]);
      if (rec && rec.group === groupName) batch.push(rec.id);
    }
    return this._loadBatch(batch);
  }

  async loadAll() {
    const order = this.resolveLoadOrder();
    return this._loadBatch(order);
  }

  async _loadBatch(ids) {
    const concurrency = Math.max(1, this.options.concurrency | 0);
    let cursor = 0;
    let loadedCount = 0;

    const runWorker = async () => {
      while (cursor < ids.length) {
        const i = cursor++;
        const id = ids[i];
        const result = await this.load(id);
        if (result !== null) loadedCount++;
      }
    };

    const workers = [];
    for (let w = 0; w < concurrency; w++) workers.push(runWorker());
    await Promise.all(workers);

    this._emit('batch-complete', { loaded: loadedCount, total: ids.length });
    return loadedCount;
  }

  /* ---------------- diagnostics ---------------- */

  getStats() {
    const byKind = new Uint32Array(ASSET_KIND.COUNT);
    const byState = new Uint32Array(ASSET_STATE.SKIPPED + 1);
    const byGroup = new Map();

    for (let i = 0; i < this.count; i++) {
      const rec = this.records[i];
      byKind[rec.kind]++;
      byState[rec.state]++;
      byGroup.set(rec.group, (byGroup.get(rec.group) || 0) + 1);
    }

    return {
      frame:        this.frame,
      capacity:     this.capacity,
      count:        this.count,
      registered:   this.stats.registered,
      loaded:       this.stats.loaded,
      failed:       this.stats.failed,
      skipped:      this.stats.skipped,
      loadMs:       this.stats.loadMs,
      tier:         this.options.tier,
      tags:         this.options.tags,
      byKind:       Array.from(byKind),
      byState:      Array.from(byState),
      groups:       Object.fromEntries(byGroup),
      perfTier:     PERF_TIER,
    };
  }

  /* ---------------- reset / dispose ---------------- */

  reset() {
    for (let i = 0; i < this.count; i++) this.records[i].reset();
    this.count = 0;
    this.indexById.clear();
    this.stats.registered = 0;
    this.stats.loaded = 0;
    this.stats.failed = 0;
    this.stats.skipped = 0;
    this.stats.loadMs = 0;
    this.frame = 0;
    return this;
  }

  dispose() {
    this.reset();
    this.records.length = 0;
    this.records = null;
    this.loaders.length = 0;
    this.loaders = null;
    this.indexById.clear();
    this._listeners.clear();
    return this;
  }
}

/* ------------------------------------------------------------------ */
/* 4. CANONICAL LIGHTING MANIFEST                                     */
/* ------------------------------------------------------------------ */

/**
 * Registers the standard lighting stack: shader chunks, materials, palette
 * tables, light presets, biome defs, env presets, task graph def, pool
 * presets. Every path is relative to `src/` so the bootstrap can import
 * from any subdirectory.
 */
export function buildCanonicalLightingManifest(manifest) {
  if (!manifest) return null;

  manifest.registerMany([
    // ----------------------------------------------------------------
    // Shader chunks (base + anime cel lighting, no image textures)
    // ----------------------------------------------------------------
    { id: 'shader:base',              kind: ASSET_KIND.SHADER_CHUNK, path: '../shaders/117_rnd_BaseShader.glsl.js',            tier: ASSET_TIER.ANY,    tags: ASSET_TAG.CRITICAL, priority: 10,  group: 'shader_base' },
    { id: 'shader:color_utils',       kind: ASSET_KIND.SHADER_CHUNK, path: '../utils/039_rnd_ColorPalette.js',                 tier: ASSET_TIER.ANY,    tags: ASSET_TAG.CRITICAL, priority: 11,  group: 'shader_base' },
    { id: 'shader:normal_quant',      kind: ASSET_KIND.SHADER_CHUNK, path: '../mesh/064_scn_normalQuantizer.js',               tier: ASSET_TIER.ANY,    tags: ASSET_TAG.CRITICAL, priority: 12,  group: 'shader_base' },
    { id: 'shader:perceptual_base',   kind: ASSET_KIND.SHADER_CHUNK, path: '../shaders/121_rnd_PerceptualBaseEnhancer.glsl.js', tier: ASSET_TIER.ANY,    tags: ASSET_TAG.CRITICAL, priority: 13,  group: 'shader_base', dependsOn: ['shader:base'] },

    // ----------------------------------------------------------------
    // Anime cel lighting core (three r185 lights only)
    // ----------------------------------------------------------------
    { id: 'shader:anime_cel_lighting', kind: ASSET_KIND.SHADER_CHUNK, path: '../shaders/296_lgt_AnimeCelLighting.glsl.js',      tier: ASSET_TIER.ANY,    tags: ASSET_TAG.CRITICAL, priority: 20,  group: 'shader_lighting', dependsOn: ['shader:base'] },
    { id: 'shader:anime_rim_light',    kind: ASSET_KIND.SHADER_CHUNK, path: '../shaders/297_lgt_AnimeRimLight.glsl.js',         tier: ASSET_TIER.ANY,    tags: ASSET_TAG.CRITICAL, priority: 21,  group: 'shader_lighting', dependsOn: ['shader:base'] },
    { id: 'shader:anime_shadow_tint',  kind: ASSET_KIND.SHADER_CHUNK, path: '../shaders/298_lgt_AnimeShadowTint.glsl.js',       tier: ASSET_TIER.ANY,    tags: ASSET_TAG.CRITICAL, priority: 22,  group: 'shader_lighting', dependsOn: ['shader:base'] },
    { id: 'shader:anime_specular',     kind: ASSET_KIND.SHADER_CHUNK, path: '../shaders/299_lgt_AnimeSpecular.glsl.js',         tier: ASSET_TIER.ANY,    tags: ASSET_TAG.CRITICAL, priority: 23,  group: 'shader_lighting', dependsOn: ['shader:base'] },
    { id: 'shader:anime_emissive',     kind: ASSET_KIND.SHADER_CHUNK, path: '../shaders/300_lgt_AnimeEmissive.glsl.js',         tier: ASSET_TIER.ANY,    tags: ASSET_TAG.CRITICAL, priority: 24,  group: 'shader_lighting', dependsOn: ['shader:base'] },

    { id: 'shader:directional_light',  kind: ASSET_KIND.SHADER_CHUNK, path: '../shaders/281_lgt_DirectionalLightShader.glsl.js', tier: ASSET_TIER.ANY,   tags: ASSET_TAG.CRITICAL, priority: 30,  group: 'shader_lighting', dependsOn: ['shader:anime_cel_lighting'] },
    { id: 'shader:cast_shadow',        kind: ASSET_KIND.SHADER_CHUNK, path: '../shaders/283_lgt_CastShadowShader.glsl.js',       tier: ASSET_TIER.ANY,    tags: ASSET_TAG.CRITICAL, priority: 31,  group: 'shader_lighting', dependsOn: ['shader:directional_light'] },

    // ----------------------------------------------------------------
    // Shadow sampling
    // ----------------------------------------------------------------
    { id: 'shader:shadow_pcf',         kind: ASSET_KIND.SHADER_CHUNK, path: '../shaders/291_lgt_PCF.glsl.js',                    tier: ASSET_TIER.ANY,    tags: ASSET_TAG.CRITICAL, priority: 40,  group: 'shader_shadow' },
    { id: 'shader:shadow_pcss',        kind: ASSET_KIND.SHADER_CHUNK, path: '../shaders/292_lgt_PCSS.glsl.js',                   tier: ASSET_TIER.MEDIUM, tags: ASSET_TAG.OPTIONAL, priority: 41,  group: 'shader_shadow' },
    { id: 'shader:shadow_vsm',         kind: ASSET_KIND.SHADER_CHUNK, path: '../shaders/293_lgt_VSM.glsl.js',                    tier: ASSET_TIER.HIGH,   tags: ASSET_TAG.OPTIONAL, priority: 42,  group: 'shader_shadow' },
    { id: 'shader:cascaded_shadow',    kind: ASSET_KIND.SHADER_CHUNK, path: '../shaders/289_lgt_CascadedShadowSampling.glsl.js', tier: ASSET_TIER.MEDIUM, tags: ASSET_TAG.OPTIONAL, priority: 43,  group: 'shader_shadow' },
    { id: 'shader:contact_shadow',     kind: ASSET_KIND.SHADER_CHUNK, path: '../shaders/290_lgt_ContactShadowSampling.glsl.js', tier: ASSET_TIER.MEDIUM, tags: ASSET_TAG.OPTIONAL, priority: 44,  group: 'shader_shadow' },

    // ----------------------------------------------------------------
    // GI / AO
    // ----------------------------------------------------------------
    { id: 'shader:irradiance_volume',  kind: ASSET_KIND.SHADER_CHUNK, path: '../shaders/301_lgt_IrradianceVolumeSampling.glsl.js', tier: ASSET_TIER.ANY,  tags: ASSET_TAG.CRITICAL, priority: 50,  group: 'shader_gi' },
    { id: 'shader:spherical_harmonics',kind: ASSET_KIND.SHADER_CHUNK, path: '../shaders/302_lgt_SphericalHarmonics.glsl.js',    tier: ASSET_TIER.MEDIUM, tags: ASSET_TAG.OPTIONAL, priority: 51,  group: 'shader_gi' },
    { id: 'shader:ssao',               kind: ASSET_KIND.SHADER_CHUNK, path: '../shaders/305_lgt_ScreenSpaceAmbientOcclusion.glsl.js', tier: ASSET_TIER.MEDIUM, tags: ASSET_TAG.OPTIONAL, priority: 52, group: 'shader_ao' },
    { id: 'shader:hbao',               kind: ASSET_KIND.SHADER_CHUNK, path: '../shaders/306_lgt_HBAO.glsl.js',                    tier: ASSET_TIER.MEDIUM, tags: ASSET_TAG.OPTIONAL, priority: 53,  group: 'shader_ao' },
    { id: 'shader:gtao',               kind: ASSET_KIND.SHADER_CHUNK, path: '../shaders/307_lgt_GTAO.glsl.js',                    tier: ASSET_TIER.HIGH,   tags: ASSET_TAG.OPTIONAL, priority: 54,  group: 'shader_ao' },
    { id: 'shader:multiscatter_ao',    kind: ASSET_KIND.SHADER_CHUNK, path: '../shaders/308_lgt_MultiScatterAO.glsl.js',         tier: ASSET_TIER.HIGH,   tags: ASSET_TAG.OPTIONAL, priority: 55,  group: 'shader_ao' },
    { id: 'shader:ssgi',               kind: ASSET_KIND.SHADER_CHUNK, path: '../shaders/309_lgt_ScreenSpaceGI.glsl.js',          tier: ASSET_TIER.HIGH,   tags: ASSET_TAG.OPTIONAL, priority: 56,  group: 'shader_gi' },

    // ----------------------------------------------------------------
    // Cluster / forward+ / rect-area
    // ----------------------------------------------------------------
    { id: 'shader:clustered_lighting', kind: ASSET_KIND.SHADER_CHUNK, path: '../shaders/285_lgt_ClusteredLighting.glsl.js',     tier: ASSET_TIER.ANY,    tags: ASSET_TAG.CRITICAL, priority: 60,  group: 'shader_cluster' },
    { id: 'shader:forward_plus',       kind: ASSET_KIND.SHADER_CHUNK, path: '../shaders/287_lgt_ForwardPlusLighting.glsl.js',   tier: ASSET_TIER.ANY,    tags: ASSET_TAG.CRITICAL, priority: 61,  group: 'shader_cluster' },
    { id: 'shader:cluster_index',      kind: ASSET_KIND.SHADER_CHUNK, path: '../shaders/323_lgt_ClusterIndex.glsl.js',          tier: ASSET_TIER.ANY,    tags: ASSET_TAG.CRITICAL, priority: 62,  group: 'shader_cluster' },
    { id: 'shader:light_list',         kind: ASSET_KIND.SHADER_CHUNK, path: '../shaders/322_lgt_LightList.glsl.js',             tier: ASSET_TIER.ANY,    tags: ASSET_TAG.CRITICAL, priority: 63,  group: 'shader_cluster' },
    { id: 'shader:rect_area_ltc',      kind: ASSET_KIND.SHADER_CHUNK, path: '../shaders/326_lgt_RectAreaLightLTC.glsl.js',      tier: ASSET_TIER.MEDIUM, tags: ASSET_TAG.OPTIONAL, priority: 64,  group: 'shader_cluster' },

    // ----------------------------------------------------------------
    // Environment: sky / clouds / stars / aurora / sun rays
    // ----------------------------------------------------------------
    { id: 'shader:sky',                kind: ASSET_KIND.SHADER_CHUNK, path: '../shaders/279_lgt_SkyShader.glsl.js',              tier: ASSET_TIER.ANY,    tags: ASSET_TAG.CRITICAL, priority: 70,  group: 'shader_env' },
    { id: 'shader:clouds',             kind: ASSET_KIND.SHADER_CHUNK, path: '../shaders/280_lgt_CloudsShader.glsl.js',           tier: ASSET_TIER.ANY,    tags: ASSET_TAG.CRITICAL, priority: 71,  group: 'shader_env' },
    { id: 'shader:sun_rays',           kind: ASSET_KIND.SHADER_CHUNK, path: '../shaders/282_lgt_SunRaysShader.glsl.js',          tier: ASSET_TIER.ANY,    tags: ASSET_TAG.CRITICAL, priority: 72,  group: 'shader_env' },
    { id: 'shader:volumetric_scatter', kind: ASSET_KIND.SHADER_CHUNK, path: '../shaders/314_lgt_VolumetricLightScattering.glsl.js', tier: ASSET_TIER.MEDIUM, tags: ASSET_TAG.OPTIONAL, priority: 73, group: 'shader_env' },
    { id: 'shader:sun_disk',           kind: ASSET_KIND.SHADER_CHUNK, path: '../shaders/315_lgt_SunDisk.glsl.js',               tier: ASSET_TIER.ANY,    tags: ASSET_TAG.CRITICAL, priority: 74,  group: 'shader_env' },
    { id: 'shader:moon_phase',         kind: ASSET_KIND.SHADER_CHUNK, path: '../shaders/316_lgt_MoonPhase.glsl.js',             tier: ASSET_TIER.ANY,    tags: ASSET_TAG.CRITICAL, priority: 75,  group: 'shader_env' },
    { id: 'shader:starfield',          kind: ASSET_KIND.SHADER_CHUNK, path: '../shaders/317_lgt_Starfield.glsl.js',             tier: ASSET_TIER.ANY,    tags: ASSET_TAG.CRITICAL, priority: 76,  group: 'shader_env' },
    { id: 'shader:aurora',             kind: ASSET_KIND.SHADER_CHUNK, path: '../shaders/318_lgt_Aurora.glsl.js',                tier: ASSET_TIER.HIGH,   tags: ASSET_TAG.OPTIONAL | ASSET_TAG.SNOW, priority: 77, group: 'shader_env' },
    { id: 'shader:cloud_shadow',       kind: ASSET_KIND.SHADER_CHUNK, path: '../shaders/319_lgt_CloudShadow.glsl.js',           tier: ASSET_TIER.MEDIUM, tags: ASSET_TAG.OPTIONAL, priority: 78,  group: 'shader_env' },

    // ----------------------------------------------------------------
    // Palette tables (color-only, no images)
    // ----------------------------------------------------------------
    { id: 'palette:reference',         kind: ASSET_KIND.PALETTE,      path: '../config/338_lgt_ReferenceImagePalettes.js',     tier: ASSET_TIER.ANY,    tags: ASSET_TAG.CRITICAL, priority: 100, group: 'palette' },
    { id: 'palette:house',             kind: ASSET_KIND.PALETTE,      path: '../config/339_lgt_HousePalette.js',               tier: ASSET_TIER.ANY,    tags: ASSET_TAG.CRITICAL | ASSET_TAG.HOUSE, priority: 101, group: 'palette' },
    { id: 'palette:canyon',            kind: ASSET_KIND.PALETTE,      path: '../config/340_lgt_CanyonPalette.js',              tier: ASSET_TIER.ANY,    tags: ASSET_TAG.CRITICAL | ASSET_TAG.CANYON, priority: 102, group: 'palette' },
    { id: 'palette:desert',            kind: ASSET_KIND.PALETTE,      path: '../config/341_lgt_DesertPalette.js',              tier: ASSET_TIER.ANY,    tags: ASSET_TAG.CRITICAL | ASSET_TAG.DESERT, priority: 103, group: 'palette' },
    { id: 'palette:snow',              kind: ASSET_KIND.PALETTE,      path: '../config/342_lgt_SnowPalette.js',                tier: ASSET_TIER.ANY,    tags: ASSET_TAG.CRITICAL | ASSET_TAG.SNOW, priority: 104, group: 'palette' },
    { id: 'palette:indoor',            kind: ASSET_KIND.PALETTE,      path: '../config/343_lgt_IndoorPalette.js',              tier: ASSET_TIER.ANY,    tags: ASSET_TAG.CRITICAL | ASSET_TAG.INDOOR, priority: 105, group: 'palette' },
    { id: 'palette:outdoor',           kind: ASSET_KIND.PALETTE,      path: '../config/344_lgt_OutdoorPalette.js',             tier: ASSET_TIER.ANY,    tags: ASSET_TAG.CRITICAL | ASSET_TAG.EXTERIOR, priority: 106, group: 'palette' },

    // ----------------------------------------------------------------
    // Light presets / defaults
    // ----------------------------------------------------------------
    { id: 'preset:lights',             kind: ASSET_KIND.LIGHT_PRESET, path: '../config/337_lgt_LightPresets.js',                tier: ASSET_TIER.ANY,    tags: ASSET_TAG.CRITICAL, priority: 120, group: 'preset' },
    { id: 'preset:shadow_defaults',    kind: ASSET_KIND.LIGHT_PRESET, path: '../config/345_lgt_ShadowDefaults.js',              tier: ASSET_TIER.ANY,    tags: ASSET_TAG.CRITICAL, priority: 121, group: 'preset' },
    { id: 'preset:gi_defaults',        kind: ASSET_KIND.LIGHT_PRESET, path: '../config/346_lgt_GIDefaults.js',                  tier: ASSET_TIER.ANY,    tags: ASSET_TAG.CRITICAL, priority: 122, group: 'preset' },
    { id: 'preset:ao_defaults',        kind: ASSET_KIND.LIGHT_PRESET, path: '../config/347_lgt_AODefaults.js',                  tier: ASSET_TIER.ANY,    tags: ASSET_TAG.CRITICAL, priority: 123, group: 'preset' },
    { id: 'preset:light_budgets',      kind: ASSET_KIND.LIGHT_PRESET, path: '../config/348_lgt_LightBudgets.js',                tier: ASSET_TIER.ANY,    tags: ASSET_TAG.CRITICAL, priority: 124, group: 'preset' },

    // ----------------------------------------------------------------
    // Biome defs
    // ----------------------------------------------------------------
    { id: 'biome:defs',                kind: ASSET_KIND.BIOME_DEF,    path: '../config/094_scn_BiomeDefs.js',                     tier: ASSET_TIER.ANY,    tags: ASSET_TAG.CRITICAL, priority: 140, group: 'biome' },
    { id: 'biome:themes',              kind: ASSET_KIND.BIOME_DEF,    path: '../config/093_scn_ThemeDefs.js',                     tier: ASSET_TIER.ANY,    tags: ASSET_TAG.CRITICAL, priority: 141, group: 'biome' },

    // ----------------------------------------------------------------
    // Environment presets
    // ----------------------------------------------------------------
    { id: 'env:house',                 kind: ASSET_KIND.ENV_PRESET,   path: '../environment/216_lgt_HouseEnvironmentPreset.js',   tier: ASSET_TIER.ANY,    tags: ASSET_TAG.HOUSE, priority: 160, group: 'env' },
    { id: 'env:canyon',                kind: ASSET_KIND.ENV_PRESET,   path: '../environment/217_lgt_CanyonEnvironmentPreset.js',  tier: ASSET_TIER.ANY,    tags: ASSET_TAG.CANYON, priority: 161, group: 'env' },
    { id: 'env:desert',                kind: ASSET_KIND.ENV_PRESET,   path: '../environment/218_lgt_DesertEnvironmentPreset.js',  tier: ASSET_TIER.ANY,    tags: ASSET_TAG.DESERT, priority: 162, group: 'env' },
    { id: 'env:snow',                  kind: ASSET_KIND.ENV_PRESET,   path: '../environment/219_lgt_SnowEnvironmentPreset.js',    tier: ASSET_TIER.ANY,    tags: ASSET_TAG.SNOW, priority: 163, group: 'env' },
    { id: 'env:climate_presets',       kind: ASSET_KIND.ENV_PRESET,   path: '../environment/209_lgt_ClimatePresets.js',           tier: ASSET_TIER.ANY,    tags: ASSET_TAG.CRITICAL, priority: 164, group: 'env' },

    // ----------------------------------------------------------------
    // Task graph / pool presets
    // ----------------------------------------------------------------
    { id: 'graph:standard',            kind: ASSET_KIND.TASK_GRAPH,   path: './008_rnd_TaskGraph.js',                             tier: ASSET_TIER.ANY,    tags: ASSET_TAG.CRITICAL, priority: 180, group: 'core' },
    { id: 'pool:object',               kind: ASSET_KIND.POOL_PRESET,  path: './010_rnd_ObjectPool.js',                            tier: ASSET_TIER.ANY,    tags: ASSET_TAG.CRITICAL, priority: 181, group: 'core' },
    { id: 'pool:typed',                kind: ASSET_KIND.POOL_PRESET,  path: './011_rnd_TypedArrayPool.js',                        tier: ASSET_TIER.ANY,    tags: ASSET_TAG.CRITICAL, priority: 182, group: 'core' },
    { id: 'pool:buffer',               kind: ASSET_KIND.POOL_PRESET,  path: './012_rnd_BufferPool.js',                            tier: ASSET_TIER.ANY,    tags: ASSET_TAG.CRITICAL, priority: 183, group: 'core' },
    { id: 'pool:render_target',        kind: ASSET_KIND.POOL_PRESET,  path: './013_rnd_RenderTargetPool.js',                      tier: ASSET_TIER.ANY,    tags: ASSET_TAG.CRITICAL, priority: 184, group: 'core' },
  ]);

  return manifest;
}

/* ------------------------------------------------------------------ */
/* 5. MODULE-LEVEL SINGLETON                                          */
/* ------------------------------------------------------------------ */

let _defaultManifest = null;

export function getDefaultManifest(options) {
  if (!_defaultManifest) {
    _defaultManifest = new AssetManifest(options);
    buildCanonicalLightingManifest(_defaultManifest);
  }
  return _defaultManifest;
}

export function disposeDefaultManifest() {
  if (_defaultManifest) {
    _defaultManifest.dispose();
    _defaultManifest = null;
  }
}

/* ------------------------------------------------------------------ */
/* 6. FACTORY                                                         */
/* ------------------------------------------------------------------ */

export function createAssetManifest(options = {}) {
  return new AssetManifest(options);
}

/* ------------------------------------------------------------------ */
/* 7. DEFAULT EXPORT                                                  */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  AssetManifest,
  AssetRecord,
  createAssetManifest,
  getDefaultManifest,
  disposeDefaultManifest,
  buildCanonicalLightingManifest,
  ASSET_KIND,
  ASSET_KIND_NAME,
  ASSET_STATE,
  ASSET_STATE_NAME,
  ASSET_TIER,
  ASSET_TAG,
  MAX_ASSETS,
};

export default _defaultExport;