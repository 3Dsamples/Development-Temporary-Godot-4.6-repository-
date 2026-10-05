API Documentation — src/core/002_rnd_App.js

File Purpose

This file is the app-level orchestration layer that sits directly on top of 001_rnd_Bootstrap.js and 008_scn_world.js. Where the Bootstrap owns the frame loop and the world singleton, the App owns:

1. The application lifecycle state machine — IDLE → BOOTING → LOADING → READY → RUNNING → PAUSED → DISPOSED.
2. DOM canvas mounting — finding the container element and attaching the renderer's canvas to it with the correct CSS for mobile touch handling.
3. The full lighting stack auto-wiring pass that dynamically imports every lighting subsystem in the manifest and registers each as a priority-ordered stage against the Bootstrap.
4. Android-only runtime guards — thermal monitoring, battery monitoring, visibility monitoring, save-data monitoring. Each guard degrades quality rather than dropping frames.
5. A small app-command surface — setTheme, setBiome, setDayCycle, setIndoorOutdoorBlend, setInterior, setExterior, toggleShadows, toggleGI, toggleAO, screenshot — that downstream directors consume.
6. Graceful degradation for missing modules — every dynamic import is wrapped in try/catch so the app boots even when a subsystem file is absent.

The design constraint is that main.js and index.html only ever import this file. Everything else in the engine imports App or the modules App already loaded.

---

Exported Constants

APP_STATE

Type: frozen enum

Values:

· IDLE — 'idle' — initial state, nothing has happened.
· BOOTING — 'booting' — boot() is running, world initializing.
· LOADING — 'loading' — lighting modules are being dynamically imported and registered.
· READY — 'ready' — everything is wired but the frame loop has not started.
· RUNNING — 'running' — the frame loop is active.
· PAUSED — 'paused' — the frame loop is scheduled but skipping update/render.
· DISPOSED — 'disposed' — full teardown has happened.

PERF_TIER

Type: string

Value: 'LOW' | 'MEDIUM' | 'HIGH' — cached from getPerfTier() at module load. Used for stage priorities and device-specific toggles.

MOBILE_DPR_CAP

Type: number

Value: 1.25 | 1.75 | 2.0 — cached device pixel-ratio cap. Any resize() call on the App clamps the requested DPR to this value.

STAGE_PRIORITY

Type: frozen object

Maps named stage slots to their priority integers. Values:

· LIGHT_MANAGER = 10
· LIGHT_SYSTEM = 20
· SHADOW_SYSTEM = 30
· GI_SYSTEM = 40
· AO_SYSTEM = 50
· ENVIRONMENT_SYSTEM = 60
· INTERIOR_SYSTEM = 70
· EXTERIOR_SYSTEM = 80
· ANIME_LIGHT_DIRECTOR = 90
· REFERENCE_LIGHT_MATCHER = 100
· COLOR_ONLY_DIRECTOR = 110

Lower values run earlier. The order encodes the dependency chain — light manager first, directors last.

LIGHTING_MODULES

Type: frozen array of { name, path, priority }

The full lighting stack manifest. Each entry describes a module the App will dynamically import and register as a stage. The entries are:

· { name: 'LightManager', path: '../lights/006_lgt_LightManager.js', priority: STAGE_PRIORITY.LIGHT_MANAGER }
· { name: 'LightingSystem', path: '../systems/267_lgt_LightingSystem.js', priority: STAGE_PRIORITY.LIGHT_SYSTEM }
· { name: 'UniversalLightManager', path: '../systems/268_lgt_UniversalLightManager.js', priority: STAGE_PRIORITY.LIGHT_SYSTEM + 1 }
· { name: 'ShadowSystem', path: '../systems/270_lgt_ShadowSystemOrchestrator.js', priority: STAGE_PRIORITY.SHADOW_SYSTEM }
· { name: 'GISystem', path: '../systems/271_lgt_GISystemOrchestrator.js', priority: STAGE_PRIORITY.GI_SYSTEM }
· { name: 'AOSystem', path: '../systems/272_lgt_AOSystemOrchestrator.js', priority: STAGE_PRIORITY.AO_SYSTEM }
· { name: 'EnvironmentSystem', path: '../systems/273_lgt_EnvironmentSystemOrchestrator.js', priority: STAGE_PRIORITY.ENVIRONMENT_SYSTEM }
· { name: 'InteriorSystem', path: '../systems/274_lgt_InteriorSystemOrchestrator.js', priority: STAGE_PRIORITY.INTERIOR_SYSTEM }
· { name: 'ExteriorSystem', path: '../systems/275_lgt_ExteriorSystemOrchestrator.js', priority: STAGE_PRIORITY.EXTERIOR_SYSTEM }
· { name: 'AnimeLightDirector', path: '../systems/276_lgt_AnimeLightDirector.js', priority: STAGE_PRIORITY.ANIME_LIGHT_DIRECTOR }
· { name: 'ReferenceLightMatcher', path: '../systems/277_lgt_ReferenceLightMatcher.js', priority: STAGE_PRIORITY.REFERENCE_LIGHT_MATCHER }
· { name: 'ColorOnlyDirector', path: '../systems/278_lgt_ColorOnlyLightDirector.js', priority: STAGE_PRIORITY.COLOR_ONLY_DIRECTOR }

---

Module-Level State (Not Exported Directly)

_defaultApp

Type: App | null

The optional module-level singleton. Created on first getDefaultApp() call. Cleared by disposeDefaultApp().

---

Exported Class — App

Constructor

```
new App(options = {})
```

Parameters (all optional):

· container — DOM element or CSS selector string. If omitted, defaults to document.body.
· autoboot — boolean; if true, boot() is called automatically in the constructor. Default true.
· autorun — boolean; if true, start() is called after boot() completes. Default true.
· width — initial viewport width. Default window.innerWidth.
· height — initial viewport height. Default window.innerHeight.
· pixelRatio — device pixel ratio, clamped to MOBILE_DPR_CAP. Default window.devicePixelRatio.
· enableShadows — boolean. Default true.
· lowPowerMode — boolean. Default false.
· thermalGuard — boolean; enable the compute-pressure thermal monitor. Default true.
· batteryGuard — boolean; enable the Battery Status API monitor. Default true.
· registerStack — boolean; if true (default), the App auto-wires the lighting stack. If false, no dynamic imports happen and the App only manages the world.

Constructor work:

1. Merges options into this.options.
2. Sets _state = APP_STATE.IDLE.
3. Allocates _loadedModules (Map), _stageEntries (Map), _listeners (Map).
4. Binds internal resize/visibility/thermal/battery handlers to instance methods so they can be removed later.
5. Initializes _stats with bootTimeMs, loadTimeMs, modulesLoaded, modulesFailed, modulesSkipped, lastError.

Instance Properties (Read-Only)

· get state — returns the current APP_STATE string.
· get isReady — true if state is READY, RUNNING, or PAUSED.
· get isRunning — true if state is RUNNING.
· get isPaused — true if state is PAUSED.
· get perfTier — cached PERF_TIER string.
· get stats — the stats object.

Instance Methods

mount(container)

Parameters: container — DOM element or selector string.

Returns: this (chainable).

Purpose: resolves the container to a DOM element and stores it in this._container. If the argument is a string, document.querySelector is used. If it resolves to null and document exists, falls back to document.body. Throws if no container is available.

boot(options = {})

Parameters: options — merged into this.options.

Returns: Promise<this>.

Purpose: the main lifecycle entry point.

Async flow:

1. If state is not IDLE, returns immediately.
2. Sets state to BOOTING.
3. Records t0 = performance.now().
4. Merges options into this.options.
5. If a container was supplied, calls mount().
6. Calls Bootstrap.boot({...}) with the app's options. This initializes the world, runs the bitECS audit, attaches listeners, and performs an initial resize.
7. Calls _attachCanvas() to add the renderer's DOM element to the container with mobile-safe CSS.
8. Calls _attachAndroidGuards() to install thermal, battery, and visibility monitors.
9. Sets state to LOADING.
10. If registerStack is not false, calls await this.registerLightingStack().
11. Wires Bootstrap events to App events via _wireBootstrapEvents().
12. Records t1 = performance.now() and stores bootTimeMs = t1 - t0.
13. Sets state to READY, emits ready.
14. If autorun is true, calls this.start().
15. Returns this.

On error: records _stats.lastError, logs, sets state back to IDLE, re-throws.

registerLightingStack()

Parameters: none.

Returns: Promise<this>.

Purpose: iterates LIGHTING_MODULES in order. For each entry:

1. Dynamically imports descriptor.path with a /* @vite-ignore */ comment so bundlers leave it alone.
2. Stores the module namespace in _loadedModules.
3. Calls _instantiateModule(mod, descriptor) to get a stage instance.
4. If an instance was returned, registers it against the Bootstrap with Bootstrap.registerStage(descriptor.name, instance, descriptor.priority) and stores the entry in _stageEntries.
5. Increments _stats.modulesLoaded or _stats.modulesSkipped.
6. On import failure, increments _stats.modulesFailed but does NOT throw. A missing module is not fatal — the engine boots in a degraded state.

After the loop, records loadTimeMs and emits stack with the final counts.

start()

Returns: this.

Purpose: calls Bootstrap.start(), sets state to RUNNING. No-op if state is not READY or PAUSED.

stop()

Returns: this.

Purpose: calls Bootstrap.stop(), sets state to READY.

pause()

Returns: this.

Purpose: calls Bootstrap.pause(), sets state to PAUSED.

resume()

Returns: this.

Purpose: calls Bootstrap.resume(), sets state to RUNNING.

resize(width, height, pixelRatio)

Parameters (all optional):

· width — if omitted, uses window.innerWidth.
· height — if omitted, uses window.innerHeight.
· pixelRatio — if omitted, uses window.devicePixelRatio.

Returns: this.

Purpose: clamps the DPR to MOBILE_DPR_CAP, updates this.options, and calls Bootstrap.resize().

dispose()

Returns: this.

Purpose: full teardown.

1. Detaches Android guards via _detachAndroidGuards().
2. Removes window and document listeners.
3. Calls Bootstrap.dispose() — which disposes every registered stage in reverse priority order, tears down the world, and resets the Bootstrap.
4. Clears _loadedModules, _stageEntries, _listeners.
5. Sets state to DISPOSED.

setTheme(themeId)

Parameters: themeId — a theme identifier string.

Returns: this.

Purpose: if the core exposes setTheme, calls it. Emits a theme event.

setBiome(biomeId, instant = false)

Parameters:

· biomeId — 0 for desert, 1 for snow, 2 for sea.
· instant — if true, skips the biome transition and snaps.

Returns: this.

Purpose: delegates to core.setBiome() if available. Emits a biome event.

setDayCycle(t)

Parameters: t — normalized [0, 1] where 0 = midnight, 0.25 = sunrise, 0.5 = noon, 0.75 = sunset.

Returns: this.

Purpose: delegates to core.setDayCycle() if available. Emits a daycycle event.

setIndoorOutdoorBlend(t)

Parameters: t — [0, 1] blend value.

Returns: this.

Purpose: if the loaded LightingSystem exposes setIndoorOutdoorBlend, calls it. Emits indooroutdoor.

setInterior(interiorId)

Parameters: interiorId — interior preset id.

Returns: this.

Purpose: delegates to the InteriorSystem if present. Emits interior.

setExterior(exteriorId)

Parameters: exteriorId — exterior preset id.

Returns: this.

Purpose: delegates to the ExteriorSystem if present. Emits exterior.

toggleShadows(enabled)

Parameters: enabled — boolean.

Returns: this.

Purpose: delegates to the ShadowSystem if present. Updates this.options.enableShadows. Emits shadows.

toggleGI(enabled)

Parameters: enabled — boolean.

Returns: this.

Purpose: delegates to the GISystem if present. Emits gi.

toggleAO(enabled)

Parameters: enabled — boolean.

Returns: this.

Purpose: delegates to the AOSystem if present. Emits ao.

screenshot(mime = 'image/png')

Parameters: mime — one of 'image/png' or 'image/jpeg'.

Returns: a data URL string, or null on failure.

Purpose: reads renderer.domElement.toDataURL(mime). Emits screenshot with the URL.

on(event, fn)

Parameters:

· event — event name string.
· fn — callback.

Returns: an unsubscribe function.

Purpose: subscribe to an App event.

off(event, fn)

Parameters:

· event — event name.
· fn — the callback to remove.

Returns: nothing.

Purpose: unsubscribe.

Internal Instance Methods (Documented)

_setState(next)

Sets _state and emits state with { from, to }. No-op if already in next.

_emit(event, payload)

Dispatches to every listener registered for event, wrapped in try/catch.

_attachCanvas()

Adds the renderer's DOM element to the container if it is not already attached. Applies the mobile-safe CSS: position: absolute, inset: 0, width: 100%, height: 100%, display: block, touch-action: none, user-select: none, -webkit-tap-highlight-color: transparent. Also sets the container to position: relative and overflow: hidden.

_instantiateModule(mod, descriptor)

Parameters:

· mod — the dynamically imported module namespace.
· descriptor — the manifest entry for the module.

Returns: the stage instance, or null.

Purpose: tries three construction conventions in order:

1. A factory named create<Name> — e.g. createLightManager.
2. A class named <Name> — e.g. LightManager.
3. A default export that is either a class or an object with an update method.

If none of those yields an instance, and the module itself has a top-level update method, uses the module namespace as the instance. This lets a module opt out of the class pattern and simply export a system object.

_wireBootstrapEvents()

Forwards every Bootstrap event (frame, resize, contextlost, contextrestored, stopped, started) to the matching App event.

_getLoadedInstance(name)

Parameters: name — the manifest name of the module.

Returns: the stage instance, or null.

Purpose: lookup helper used by all the setX/toggleX commands.

_attachAndroidGuards()

Installs three guards:

1. Window resize and orientationchange listeners.
2. Document visibilitychange listener that calls pause() when hidden and resume() when visible.
3. A compute-pressure observer (Chrome-only API) that reports thermal states.
4. A Battery Status API listener that watches level and charging state.

_detachAndroidGuards()

Removes the listeners installed by _attachAndroidGuards. Also calls disconnect() on the pressure observer and removes the battery listeners.

_onAppResize()

Debounced window resize handler. Calls this.resize() with current dimensions and clamped DPR.

_onVisibility()

Called on visibilitychange. Pauses when hidden, resumes when visible.

_onThermalChange(state)

Parameters: state — either a string 'nominal' | 'fair' | 'serious' | 'critical' or an object { state }.

Purpose: when thermal state is serious or critical, disables shadows and GI. When it returns to nominal, restores the previous settings.

_onBatteryChange()

Purpose: reads _battery.level and _battery.charging. If unplugged and below 15 %, enters low-power mode — shadows, GI, and AO are disabled. If charging or above 25 %, exits low-power mode.

---

Exported Functions

createApp(options = {})

Parameters: options — same shape as the App constructor.

Returns: a new App instance.

Purpose: pure factory. Use this instead of new App() when you want to avoid the constructor side effects that getDefaultApp would trigger.

getDefaultApp()

Parameters: none.

Returns: the module-level singleton App instance, creating it on first call.

Purpose: convenience accessor for downstream code that wants to reach the App's commands without holding a reference.

disposeDefaultApp()

Parameters: none.

Returns: nothing.

Purpose: disposes and clears the module-level singleton.

---

Default Export

The default export is a single frozen object bundling App, createApp, getDefaultApp, disposeDefaultApp, APP_STATE, LIGHTING_MODULES, and STAGE_PRIORITY.

---

Usage Pattern

main.js:

```
import App, { APP_STATE } from './src/core/002_rnd_App.js';

const app = new App({
  container: '#app',
  autoboot: true,
  autorun: true,
  enableShadows: true,
});

app.on('ready', () => {
  console.log('Engine ready');
  app.setBiome(0);
  app.setDayCycle(0.38);
});

app.on('thermal', ({ state }) => {
  console.log('Thermal state:', state);
});

app.on('battery', ({ level, charging }) => {
  console.log('Battery:', level, charging);
});
```

The App boot flow is entirely deterministic. Given the same options and the same availability of downstream modules, the sequence of state transitions and the order of stage registrations is identical. That makes it safe for regression tests to snapshot the loaded stack and compare against a reference.

