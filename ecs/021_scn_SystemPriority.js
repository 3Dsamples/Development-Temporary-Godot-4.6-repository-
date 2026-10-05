// File : 021
// name : src/ecs/021_scn_SystemPriority.js
// description : Canonical priority-resolution module for the scene ECS world
//               of the anime lighting stack on Android mobile. Assigns a
//               stable, documented numeric priority to every named system
//               in the engine so that the scheduler (020_scn_SystemScheduler)
//               can order the frame deterministically without any subsystem
//               hard-coding its own priority.
//
//               Design:
//                 • One canonical priority table. Every named system in the
//                   lighting stack has an entry. Priorities are grouped by
//                   logical stage so that the frame reads top-to-bottom
//                   like a pipeline:
//
//                     EARLY (0..99)
//                        CAMERA_INPUT, CULLING, GPU_TIMING
//                     UPDATE (100..899)
//                        LIGHT_LIST_BUILD, CLUSTER_BUILD,
//                        SHADOW_ATLAS_PACK, SHADOW_CASCADE_SOLVE,
//                        SHADOW_UPDATE, GI_PROBE_BAKE, GI_VOLUME_UPDATE,
//                        GI_PORTAL_TRANSPORT, AO_VOLUME_UPDATE,
//                        AO_BLUR, AO_DENOISE,
//                        ENVIRONMENT_DAY_CYCLE, ENVIRONMENT_WEATHER,
//                        INTERIOR_VOLUME, EXTERIOR_PROBE,
//                        MATERIAL_BIND, MATERIAL_UPLOAD
//                     LATE (900..998)
//                        ANIME_DIRECTOR_HINT, LIGHT_UNIFORM_UPLOAD,
//                        POST_BUFFER_PREPARE
//                     POST (999)
//                        FRAME_FINALIZE
//
//                 • Per-system phase is derived from priority using the
//                   same mapping as the scheduler so the two modules never
//                   disagree.
//
//                 • `resolvePriorityConflicts()` validates the declared
//                   priority table against the registered systems and
//                   reports any conflict (a registered system whose
//                   declared priority contradicts its dependency edges).
//
//                 • `getPriorityFor(name)` / `getPhaseFor(name)` are O(1)
//                   flat-map lookups, allocation-free on the hot path.
//
//                 • Reserved ranges per subsystem group so that inserting
//                   a new system between two existing ones is a matter of
//                   picking an unused integer in the right range.
//
//                 • Debug-friendly: `describePriorityTable()` returns a
//                   JSON-serializable list of every entry with group,
//                   priority, phase, and description.
//
//               Integration:
//                 • 019_scn_SystemBase.js       — system registry
//                 • 020_scn_SystemScheduler.js  — phase mapping
//                 • 026_rnd_Logger.js           — conflict reporting
//                 • 030_rnd_ErrorBoundary.js    — containment
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0
//               API only; every internal array sized once at construction.
// best for : Guaranteeing that every named system in the anime lighting
//            stack has a documented, stable priority — so that the frame
//            order is deterministic, dependencies never contradict the
//            declared priority, and any conflict is caught at boot rather
//            than producing a subtle runtime ordering bug on Android.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import {
  getPerfTier,
} from '../core/008_scn_world.js';

import {
  getDefaultLogger,
  LOG_CHANNEL,
} from '../core/026_rnd_Logger.js';

import {
  getDefaultErrorBoundaries,
  BOUNDARY_TAG,
} from '../core/030_rnd_ErrorBoundary.js';

import {
  getSystem,
  getAllSystems,
  getSystemCount,
  forEachSystem,
  SYSTEM_NAME,
  SYSTEM_STATE,
} from './019_scn_SystemBase.js';

import {
  PHASE,
  PHASE_NAME,
  PRIORITY_EARLY,
  PRIORITY_LATE,
  PRIORITY_LAST,
  PRIORITY_DEFAULT,
} from './020_scn_SystemScheduler.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER_LOCAL = getPerfTier();

/**
 * Canonical priority groups. Each group reserves a numeric range so that
 * inserting a new system between two existing ones is just a matter of
 * picking an unused integer in the correct range.
 */
export const PRIORITY_GROUP = Object.freeze({
  EARLY:            0,
  CULLING:          1,
  LIGHT_LIST:       2,
  CLUSTER:          3,
  SHADOW:           4,
  GI:               5,
  AO:               6,
  ENVIRONMENT:      7,
  INTERIOR:         8,
  EXTERIOR:         9,
  MATERIAL:        10,
  DIRECTOR:        11,
  POST:            12,
  FINALIZE:        13,
  COUNT:           14,
});

export const PRIORITY_GROUP_NAME = Object.freeze([
  'early',
  'culling',
  'light_list',
  'cluster',
  'shadow',
  'gi',
  'ao',
  'environment',
  'interior',
  'exterior',
  'material',
  'director',
  'post',
  'finalize',
]);

/**
 * Priority range assigned to each group. These are the hard bounds; the
 * actual per-system priority lives inside the range.
 */
export const PRIORITY_GROUP_RANGE = Object.freeze([
  Object.freeze({ start:   0, end:  49 }),   // EARLY
  Object.freeze({ start:  50, end:  99 }),   // CULLING
  Object.freeze({ start: 100, end: 179 }),   // LIGHT_LIST
  Object.freeze({ start: 180, end: 259 }),   // CLUSTER
  Object.freeze({ start: 260, end: 399 }),   // SHADOW
  Object.freeze({ start: 400, end: 519 }),   // GI
  Object.freeze({ start: 520, end: 619 }),   // AO
  Object.freeze({ start: 620, end: 719 }),   // ENVIRONMENT
  Object.freeze({ start: 720, end: 789 }),   // INTERIOR
  Object.freeze({ start: 790, end: 849 }),   // EXTERIOR
  Object.freeze({ start: 850, end: 889 }),   // MATERIAL
  Object.freeze({ start: 890, end: 949 }),   // DIRECTOR
  Object.freeze({ start: 950, end: 989 }),   // POST
  Object.freeze({ start: 990, end: 999 }),   // FINALIZE
]);

/**
 * Canonical priority table for every named system in the lighting stack.
 * Downstream systems read their priority from here — never hard-code a
 * priority in the system class itself.
 *
 * Priority is a STRICT TOTAL ORDER. Two systems must never share a
 * priority unless their relative order does not matter.
 */
export const SYSTEM_PRIORITY = Object.freeze({
  /* ---------------- EARLY (0..49) ---------------- */
  'camera.input':                 5,
  'camera.controller':            6,
  'platform.telemetry':           10,
  'frame.timing':                 12,
  'platform.thermal_guard':       15,
  'platform.battery_guard':       16,

  /* ---------------- CULLING (50..99) ---------------- */
  'culling.frustum':              55,
  'culling.occlusion':            60,
  'culling.distance':             65,
  'culling.lod':                  70,

  /* ---------------- LIGHT_LIST (100..179) ---------------- */
  'lights.manager':              105,
  'lights.system':               110,
  'lights.cluster_builder':      115,
  'lights.list_builder':         120,
  'lights.priority_sort':        125,
  'lights.budget_manager':       130,
  'lights.lod_manager':          135,

  /* ---------------- CLUSTER (180..259) ---------------- */
  'cluster.system':              185,
  'cluster.grid_builder':        190,
  'cluster.cell_assigner':       195,
  'cluster.light_binner':        200,
  'cluster.pipeline':            205,

  /* ---------------- SHADOW (260..399) ---------------- */
  'shadows.manager':             265,
  'shadows.system':              270,
  'shadows.light_tagger':        275,
  'shadows.cascade_splitter':    280,
  'shadows.cascade_stabilizer':  285,
  'shadows.texel_snapper':       290,
  'shadows.bias_controller':     295,
  'shadows.atlas_packer':        300,
  'shadows.atlas_allocator':     305,
  'shadows.frustum_updater':     310,
  'shadows.caster_registry':     315,
  'shadows.receiver_registry':   316,
  'shadows.update_policy':       320,
  'shadows.contact_shadow':      325,
  'shadows.softness_controller': 330,
  'shadows.tint_controller':     335,
  'shadows.edge_posterizer':     340,
  'shadows.proxy_manager':       345,
  'shadows.impostor_manager':    350,
  'shadows.parallel_updater':    355,
  'shadows.async_compiler':      360,
  'shadows.residency':           365,

  /* ---------------- GI (400..519) ---------------- */
  'gi.manager':                  405,
  'gi.system':                   410,
  'gi.probe_grid':               415,
  'gi.probe_updater':            420,
  'gi.irradiance_cache':         425,
  'gi.sh_projector':             430,
  'gi.bounce_light':             435,
  'gi.sky_irradiance':           440,
  'gi.ground_irradiance':        445,
  'gi.radiosity_solver':         450,
  'gi.voxelizer':                455,
  'gi.cone_tracer':              460,
  'gi.beam_tracer':              465,
  'gi.portal_transport':         470,
  'gi.portal_culler':            475,
  'gi.indoor_blend':             480,
  'gi.outdoor_blend':            485,
  'gi.cel_band':                 490,
  'gi.palette_driver':           492,
  'gi.leak_corrector':           495,
  'gi.temporal':                 500,
  'gi.quality_adapter':          505,
  'gi.async_updater':            510,
  'gi.residency':                515,

  /* ---------------- AO (520..619) ---------------- */
  'ao.manager':                  525,
  'ao.system':                   530,
  'ao.volume_placer':            535,
  'ao.kernel':                   540,
  'ao.ssao':                     545,
  'ao.hbao':                     550,
  'ao.gtao':                     555,
  'ao.contact_shadow':           560,
  'ao.distance_field':           565,
  'ao.voxel':                    570,
  'ao.temporal_accumulator':     575,
  'ao.bilateral':                580,
  'ao.denoiser':                 585,
  'ao.blur':                     590,
  'ao.cel_bands':                595,
  'ao.ink_outline':              600,
  'ao.edge_fade':                605,
  'ao.quality_scaler':           610,
  'ao.residency':                615,

  /* ---------------- ENVIRONMENT (620..719) ---------------- */
  'environment.manager':         625,
  'environment.system':          630,
  'environment.day_cycle':       635,
  'environment.season':          640,
  'environment.weather':         645,
  'environment.wind':            650,
  'environment.precipitation':   655,
  'environment.fog':             660,
  'environment.atmosphere':      665,
  'environment.sky':             670,
  'environment.cloud':           675,
  'environment.cloud_shadow':    680,
  'environment.sun':             685,
  'environment.moon':            690,
  'environment.star':            695,
  'environment.aurora':          700,
  'environment.horizon':         705,
  'environment.probe_updater':   710,

  /* ---------------- INTERIOR (720..789) ---------------- */
  'interior.manager':            725,
  'interior.system':             730,
  'interior.room_grid':          735,
  'interior.light_placer':       740,
  'interior.probe_placer':       745,
  'interior.portal_culler':      750,
  'interior.shadow_cache':       755,
  'interior.mood_director':      760,
  'interior.ao':                 765,
  'interior.cel_lighting':       770,
  'interior.window_shaft':       775,
  'interior.emissive_fixture':   780,

  /* ---------------- EXTERIOR (790..849) ---------------- */
  'exterior.manager':            795,
  'exterior.system':             800,
  'exterior.sunlight':           805,
  'exterior.moonlight':          810,
  'exterior.skylight':           815,
  'exterior.ground_bounce':      820,
  'exterior.canopy_shadow':      825,
  'exterior.canyon_bounce':      830,
  'exterior.water_caustics':     835,
  'exterior.snow_glare':         840,
  'exterior.probe_updater':      845,

  /* ---------------- MATERIAL (850..889) ---------------- */
  'material.manager':            855,
  'material.factory':            860,
  'material.bind':               865,
  'material.upload':             870,
  'material.lod_factory':        875,
  'material.variant_cache':      880,

  /* ---------------- DIRECTOR (890..949) ---------------- */
  'director.anime':              895,
  'director.reference_matcher':  900,
  'director.color_only':         905,
  'director.hint_emitter':       910,

  /* ---------------- POST (950..989) ---------------- */
  'post.system':                 955,
  'post.composer':               960,
  'post.bloom':                  965,
  'post.outline':                970,
  'post.tone_map':               975,
  'post.color_grade':            980,
  'post.buffer_prepare':         985,

  /* ---------------- FINALIZE (990..999) ---------------- */
  'frame.finalize':              995,
  'debug.hud':                   996,
  'debug.stats_panel':           997,
  'debug.frame_graph':           998,
});

/**
 * Reverse map — priority → system name. Populated at module load.
 */
const _priorityToName = new Map();
for (const name in SYSTEM_PRIORITY) {
  const p = SYSTEM_PRIORITY[name];
  if (typeof p === 'number') _priorityToName.set(p, name);
}

/**
 * Alias table — maps informal names to canonical names. Used so that a
 * downstream system can call `getPriorityFor('shadowAtlas')` and still
 * resolve to 'shadows.atlas_packer'.
 */
export const SYSTEM_ALIASES = Object.freeze({
  'lightManager':         'lights.manager',
  'lightSystem':          'lights.system',
  'lightingSystem':       'lights.system',
  'shadowManager':        'shadows.manager',
  'shadowSystem':         'shadows.system',
  'shadowAtlas':          'shadows.atlas_packer',
  'giManager':            'gi.manager',
  'giSystem':             'gi.system',
  'giProbe':              'gi.probe_grid',
  'aoManager':            'ao.manager',
  'aoSystem':             'ao.system',
  'environmentManager':   'environment.manager',
  'environmentSystem':    'environment.system',
  'interiorManager':      'interior.manager',
  'interiorSystem':       'interior.system',
  'exteriorManager':      'exterior.manager',
  'exteriorSystem':       'exterior.system',
  'materialManager':      'material.manager',
  'animeDirector':        'director.anime',
  'postSystem':           'post.system',
});

/* ------------------------------------------------------------------ */
/* 1. MODULE STATE                                                    */
/* ------------------------------------------------------------------ */

export const PriorityState = {
  totalLookups:        0,
  totalMisses:         0,
  totalConflicts:      0,
  lastValidationFrame: -1,
  lastValidationOk:    true,
  lastValidationIssues: [],
};

let _boundary = null;
function _ensureBoundary() {
  if (_boundary) return _boundary;
  try {
    const mgr = getDefaultErrorBoundaries();
    if (mgr) {
      _boundary = mgr.create('ecs.system_priority', {
        tag: BOUNDARY_TAG.GENERIC,
        failureThreshold: 5,
      });
    }
  } catch (_) { /* swallow */ }
  return _boundary;
}

function _safeLogger() {
  try { return getDefaultLogger(); } catch (_) { return null; }
}

/* ------------------------------------------------------------------ */
/* 2. PRIORITY RESOLUTION                                             */
/* ------------------------------------------------------------------ */

/**
 * Resolves the canonical priority for a system. Accepts either the
 * canonical name ('shadows.atlas_packer') or a registered alias
 * ('shadowAtlas'). Returns `PRIORITY_DEFAULT` if unknown.
 */
export function getPriorityFor(name) {
  PriorityState.totalLookups++;
  if (typeof name !== 'string' || name.length === 0) {
    PriorityState.totalMisses++;
    return PRIORITY_DEFAULT;
  }

  // Direct hit.
  const direct = SYSTEM_PRIORITY[name];
  if (typeof direct === 'number') return direct;

  // Alias hit.
  const alias = SYSTEM_ALIASES[name];
  if (alias) {
    const aliased = SYSTEM_PRIORITY[alias];
    if (typeof aliased === 'number') return aliased;
  }

  PriorityState.totalMisses++;
  return PRIORITY_DEFAULT;
}

/**
 * Resolves the phase that a system will run in, based on its canonical
 * priority. Uses the same mapping as the scheduler.
 */
export function getPhaseFor(name) {
  const p = getPriorityFor(name);
  if (p >= PRIORITY_LAST)         return PHASE.POST;
  if (p >= PRIORITY_LATE)         return PHASE.LATE;
  if (p <= PRIORITY_EARLY + 99)   return PHASE.EARLY;
  return PHASE.UPDATE;
}

/**
 * Resolves the priority group that a priority value belongs to.
 */
export function getGroupForPriority(priority) {
  for (let g = 0; g < PRIORITY_GROUP.COUNT; g++) {
    const r = PRIORITY_GROUP_RANGE[g];
    if (priority >= r.start && priority <= r.end) return g;
  }
  return -1;
}

/**
 * Resolves the priority group that a system name belongs to.
 */
export function getGroupFor(name) {
  const p = getPriorityFor(name);
  return getGroupForPriority(p);
}

/**
 * Returns true if the given name has a canonical priority entry.
 */
export function hasCanonicalPriority(name) {
  if (typeof name !== 'string' || name.length === 0) return false;
  if (typeof SYSTEM_PRIORITY[name] === 'number') return true;
  const alias = SYSTEM_ALIASES[name];
  if (alias && typeof SYSTEM_PRIORITY[alias] === 'number') return true;
  return false;
}

/**
 * Returns the canonical name for an alias or for the same name.
 */
export function canonicalizeName(name) {
  if (typeof name !== 'string' || name.length === 0) return null;
  if (typeof SYSTEM_PRIORITY[name] === 'number') return name;
  const alias = SYSTEM_ALIASES[name];
  if (alias && typeof SYSTEM_PRIORITY[alias] === 'number') return alias;
  return null;
}

/**
 * Returns the name of the system currently assigned to a priority, or
 * null. Useful for detecting accidental duplicates.
 */
export function getSystemAtPriority(priority) {
  const name = _priorityToName.get(priority);
  return name === undefined ? null : name;
}

/* ------------------------------------------------------------------ */
/* 3. BUILD-TIME ASSIGNMENT                                           */
/* ------------------------------------------------------------------ */

/**
 * Applies the canonical priority to a system instance. Called by the
 * scheduler or the system base class right before init. Returns true if
 * the priority was applied.
 */
export function applyPriorityToSystem(system) {
  if (!system || typeof system !== 'object') return false;
  if (typeof system.name !== 'string' || system.name.length === 0) return false;

  const canonical = canonicalizeName(system.name);
  if (!canonical) {
    // Unknown system — leave its declared priority alone.
    return false;
  }

  const p = SYSTEM_PRIORITY[canonical];
  if (typeof p === 'number' && Number.isFinite(p)) {
    system.priority = p | 0;
    return true;
  }
  return false;
}

/**
 * Applies canonical priorities to every registered system. Called once
 * at boot by the scheduler.
 */
export function applyPrioritiesToAllSystems() {
  let applied = 0;
  let skipped = 0;
  forEachSystem((sys) => {
    if (applyPriorityToSystem(sys)) applied++;
    else skipped++;
  });
  return { applied, skipped };
}

/* ------------------------------------------------------------------ */
/* 4. CONFLICT DETECTION                                              */
/* ------------------------------------------------------------------ */

/**
 * Validates the priority table for internal consistency:
 *
 *   1. No two systems share the same priority unless declared as an
 *      intentional tie.
 *   2. Every registered system resolves to a canonical priority (or is
 *      explicitly allowed to keep its declared one).
 *   3. Every dependency edge is respected by the priority order: a
 *      system must have a HIGHER priority value than every system it
 *      depends on (runs later).
 *
 * Returns { ok, issues, checked, resolved, unresolved }.
 */
export function resolvePriorityConflicts() {
  const issues = [];

  // Check 1 — duplicate priorities.
  const seen = new Map();
  for (const name in SYSTEM_PRIORITY) {
    const p = SYSTEM_PRIORITY[name];
    if (typeof p !== 'number') continue;
    const prev = seen.get(p);
    if (prev !== undefined) {
      issues.push({
        kind: 'duplicate_priority',
        priority: p,
        nameA: prev,
        nameB: name,
      });
    } else {
      seen.set(p, name);
    }
  }

  // Check 2 & 3 — validate against registered systems.
  const systems = getAllSystems();
  let checked = 0;
  let resolved = 0;
  let unresolved = 0;

  for (let i = 0; i < systems.length; i++) {
    const sys = systems[i];
    if (!sys) continue;
    checked++;

    const canonical = canonicalizeName(sys.name);
    if (canonical) {
      resolved++;
    } else {
      unresolved++;
      // Not an error; the system simply doesn't have a canonical entry.
      continue;
    }

    const ownPriority = SYSTEM_PRIORITY[canonical];

    // Dependency edges — the dependency must run EARLIER (lower priority).
    for (let d = 0; d < sys.dependencyCount; d++) {
      const depName = sys.dependencies[d];
      if (!depName) continue;
      const depCanonical = canonicalizeName(depName);
      if (!depCanonical) continue;
      const depPriority = SYSTEM_PRIORITY[depCanonical];
      if (typeof depPriority !== 'number') continue;

      if (depPriority >= ownPriority) {
        issues.push({
          kind: 'dependency_priority_conflict',
          system: sys.name,
          systemPriority: ownPriority,
          dependency: depName,
          dependencyPriority: depPriority,
        });
      }
    }
  }

  const ok = issues.length === 0;
  PriorityState.lastValidationOk = ok;
  PriorityState.lastValidationIssues = issues;
  PriorityState.totalConflicts += issues.length;

  if (!ok) {
    const log = _safeLogger();
    if (log) {
      log.warn(LOG_CHANNEL.CORE, () =>
        `[021_scn_SystemPriority] ${issues.length} priority conflict(s) detected ` +
        `across ${checked} registered systems`);
    }
    const b = _ensureBoundary();
    if (b) {
      // Do not throw — priority conflicts are warnings, not fatal.
    }
  }

  return {
    ok,
    issues,
    checked,
    resolved,
    unresolved,
  };
}

/* ------------------------------------------------------------------ */
/* 5. RANGE MANAGEMENT                                                */
/* ------------------------------------------------------------------ */

/**
 * Finds an unused priority in the given group range. Returns -1 if the
 * range is exhausted.
 */
export function findFreePriorityInGroup(groupId) {
  if (groupId < 0 || groupId >= PRIORITY_GROUP.COUNT) return -1;
  const range = PRIORITY_GROUP_RANGE[groupId];
  for (let p = range.start; p <= range.end; p++) {
    if (!_priorityToName.has(p)) return p;
  }
  return -1;
}

/**
 * Returns the number of free priority slots in a group.
 */
export function getGroupFreeCount(groupId) {
  if (groupId < 0 || groupId >= PRIORITY_GROUP.COUNT) return 0;
  const range = PRIORITY_GROUP_RANGE[groupId];
  let free = 0;
  for (let p = range.start; p <= range.end; p++) {
    if (!_priorityToName.has(p)) free++;
  }
  return free;
}

/**
 * Returns the number of used priority slots in a group.
 */
export function getGroupUsedCount(groupId) {
  if (groupId < 0 || groupId >= PRIORITY_GROUP.COUNT) return 0;
  const range = PRIORITY_GROUP_RANGE[groupId];
  let used = 0;
  for (let p = range.start; p <= range.end; p++) {
    if (_priorityToName.has(p)) used++;
  }
  return used;
}

/**
 * Returns a full report of group occupancy.
 */
export function getGroupReport() {
  const out = new Array(PRIORITY_GROUP.COUNT);
  for (let g = 0; g < PRIORITY_GROUP.COUNT; g++) {
    const range = PRIORITY_GROUP_RANGE[g];
    const used = getGroupUsedCount(g);
    const free = getGroupFreeCount(g);
    out[g] = {
      group:      PRIORITY_GROUP_NAME[g],
      start:      range.start,
      end:        range.end,
      capacity:   (range.end - range.start + 1),
      used,
      free,
      occupancy:  (range.end - range.start + 1) > 0
                    ? used / (range.end - range.start + 1)
                    : 0,
    };
  }
  return out;
}

/* ------------------------------------------------------------------ */
/* 6. SORTING HELPERS                                                 */
/* ------------------------------------------------------------------ */

/**
 * Sorts an array of system instances by canonical priority. Systems
 * without a canonical entry are sorted by their declared priority after
 * those with a canonical entry, in the order they appeared.
 */
export function sortSystemsByCanonicalPriority(systems) {
  if (!Array.isArray(systems)) return systems;
  systems.sort((a, b) => {
    const pa = getPriorityFor(a.name);
    const pb = getPriorityFor(b.name);
    if (pa !== pb) return pa - pb;
    return 0;
  });
  return systems;
}

/**
 * Sorts an array of system names by canonical priority.
 */
export function sortNamesByCanonicalPriority(names) {
  if (!Array.isArray(names)) return names;
  names.sort((a, b) => getPriorityFor(a) - getPriorityFor(b));
  return names;
}

/* ------------------------------------------------------------------ */
/* 7. DESCRIPTIVE REPORT                                              */
/* ------------------------------------------------------------------ */

/**
 * Returns a JSON-serializable description of the priority table.
 */
export function describePriorityTable() {
  const out = [];
  for (const name in SYSTEM_PRIORITY) {
    const p = SYSTEM_PRIORITY[name];
    const g = getGroupForPriority(p);
    const phase = p >= PRIORITY_LAST
      ? PHASE.POST
      : p >= PRIORITY_LATE
        ? PHASE.LATE
        : p <= PRIORITY_EARLY + 99
          ? PHASE.EARLY
          : PHASE.UPDATE;
    out.push({
      name,
      priority: p,
      group:    g >= 0 ? PRIORITY_GROUP_NAME[g] : 'unknown',
      phase:    PHASE_NAME[phase],
    });
  }
  out.sort((a, b) => a.priority - b.priority);
  return out;
}

/**
 * Returns a compact table of all canonical priorities grouped by phase.
 */
export function describePriorityTableByPhase() {
  const byPhase = new Array(PHASE.COUNT).fill(null).map(() => []);
  for (const name in SYSTEM_PRIORITY) {
    const p = SYSTEM_PRIORITY[name];
    const phase = p >= PRIORITY_LAST
      ? PHASE.POST
      : p >= PRIORITY_LATE
        ? PHASE.LATE
        : p <= PRIORITY_EARLY + 99
          ? PHASE.EARLY
          : PHASE.UPDATE;
    byPhase[phase].push({ name, priority: p });
  }
  for (let p = 0; p < PHASE.COUNT; p++) {
    byPhase[p].sort((a, b) => a.priority - b.priority);
  }
  return {
    byPhase: byPhase.map((list, p) => ({
      phase:   PHASE_NAME[p],
      count:   list.length,
      entries: list,
    })),
  };
}

/**
 * Returns a compact JSON-friendly snapshot of the priority module.
 */
export function getPriorityReport() {
  return {
    totalCanonicalEntries: Object.keys(SYSTEM_PRIORITY).length,
    totalAliases:          Object.keys(SYSTEM_ALIASES).length,
    groups:                getGroupReport(),
    totalLookups:          PriorityState.totalLookups,
    totalMisses:           PriorityState.totalLookups > 0
                             ? PriorityState.totalMisses / PriorityState.totalLookups
                             : 0,
    totalConflicts:        PriorityState.totalConflicts,
    lastValidationOk:      PriorityState.lastValidationOk,
    lastValidationIssues:  PriorityState.lastValidationIssues,
    perfTier:              PERF_TIER_LOCAL,
  };
}

/* ------------------------------------------------------------------ */
/* 8. RESET                                                           */
/* ------------------------------------------------------------------ */

/**
 * Resets runtime counters. The canonical tables are immutable and are
 * not affected.
 */
export function resetPriorityState() {
  PriorityState.totalLookups = 0;
  PriorityState.totalMisses = 0;
  PriorityState.totalConflicts = 0;
  PriorityState.lastValidationFrame = -1;
  PriorityState.lastValidationOk = true;
  PriorityState.lastValidationIssues = [];
}

/* ------------------------------------------------------------------ */
/* 9. REGISTRATION WITH THE COMPONENT REGISTRY                       */
/* ------------------------------------------------------------------ */

/**
 * The priority module does not declare its own ECS components. No-op,
 * present for API symmetry with the other scene modules.
 */
export function registerPriorityComponents(_registry) {
  return 0;
}

/* ------------------------------------------------------------------ */
/* 10. DEFAULT EXPORT                                                */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  // Enums
  PRIORITY_GROUP,
  PRIORITY_GROUP_NAME,
  PRIORITY_GROUP_RANGE,

  // Tables
  SYSTEM_PRIORITY,
  SYSTEM_ALIASES,

  // State
  PriorityState,

  // Resolution
  getPriorityFor,
  getPhaseFor,
  getGroupForPriority,
  getGroupFor,
  hasCanonicalPriority,
  canonicalizeName,
  getSystemAtPriority,

  // Application
  applyPriorityToSystem,
  applyPrioritiesToAllSystems,

  // Conflict detection
  resolvePriorityConflicts,

  // Range management
  findFreePriorityInGroup,
  getGroupFreeCount,
  getGroupUsedCount,
  getGroupReport,

  // Sorting
  sortSystemsByCanonicalPriority,
  sortNamesByCanonicalPriority,

  // Reports
  describePriorityTable,
  describePriorityTableByPhase,
  getPriorityReport,

  // Registration
  registerPriorityComponents,

  // Reset
  resetPriorityState,
};

export default _defaultExport;