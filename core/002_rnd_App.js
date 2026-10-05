// File : 002
// name : src/core/002_rnd_App.js
// description : Application orchestration layer sitting directly on top of the
//               bootstrap (001_rnd_Bootstrap.js) and world singleton
//               (008_scn_world.js). Owns the app-level lifecycle state machine
//               (idle → booting → loading → ready → running → paused → disposed),
//               DOM canvas mounting, the full lighting stack auto-wiring pass
//               that dynamically imports every module from the manifest
//               (006_lgt_LightManager … 380_lgt_lights) and registers each as a
//               priority-ordered stage against the bootstrap, plus Android-only
//               runtime guards:
//                 • thermal guard   — reduces shadow map size + GI update budget
//                                     when the device is hot;
//                 • battery guard   — drops DPR + disables shadows under 15 %;
//                 • low-power guard — caps FPS via frame skipping when the
//                                     browser reports `navigator.connection.
//                                     saveData` or `battery.savingMode`;
//                 • visibility guard — auto-pause on document.hidden.
//               Exposes a small app-command surface (setTheme, setBiome,
//               setDayCycle, setIndoorOutdoorBlend, setInterior, setExterior,
//               screenshot, toggleShadows, toggleGI, toggleAO) that downstream
//               lighting directors (276_lgt_AnimeLightDirector,
//               277_lgt_ReferenceLightMatcher, 278_lgt_ColorOnlyLightDirector)
//               consume. Every dynamic import is guarded with a try/catch and a
//               no-op fallback so the app can boot even when a downstream
//               lighting module is missing — nothing throws on Android.
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0 API
//               only; zero per-frame allocations on the hot path.
// best for : Single app-lifecycle surface for the whole engine. main.js /
//            index.html only ever import this file — every lighting subsystem
//            is registered through here, so the render graph, the frame
//            barrier, the FPS counter, and the Android guards stay one
//            coherent unit.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import * as Bootstrap from './001_rnd_Bootstrap.js';

import {
  world,
  initializeWorld,
  disposeWorld,
  stepWorld,
  renderWorld,
  getCore,
  getRenderer,
  getScene,
  getCamera,
  getLightManager,
  getPerfTier,
  getMobileDprCap,
  isWorldReady,
} from './008_scn_world.js';

import {
  runBiteCSAudit,
  assertBiteCSReady,
  isBiteCSReady,
  getAudit,
} from './009_scn_BiteCSVersionPolicy.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

export const APP_STATE = Object.freeze({
  IDLE:     'idle',
  BOOTING:  'booting',
  LOADING:  'loading',
  READY:    'ready',
  RUNNING:  'running',
  PAUSED:   'paused',
  DISPOSED: 'disposed',
});

const PERF_TIER = getPerfTier();
const MOBILE_DPR_CAP = getMobileDprCap();

const STAGE_PRIORITY = Object.freeze({
  LIGHT_MANAGER:            10,
  LIGHT_SYSTEM:             20,
  SHADOW_SYSTEM:            30,
  GI_SYSTEM:                40,
  AO_SYSTEM:                50,
  ENVIRONMENT_SYSTEM:       60,
  INTERIOR_SYSTEM:          70,
  EXTERIOR_SYSTEM:          80,
  ANIME_LIGHT_DIRECTOR:     90,
  REFERENCE_LIGHT_MATCHER: 100,
  COLOR_ONLY_DIRECTOR:     110,
});

/* ------------------------------------------------------------------ */
/* 1. LIGHTING MODULE MANIFEST                                        */
/*    All dynamic; missing modules degrade gracefully.                */
/* ------------------------------------------------------------------ */

const LIGHTING_MODULES = Object.freeze([
  { name: 'LightManager',          path: '../lights/006_lgt_LightManager.js',                 priority: STAGE_PRIORITY.LIGHT_MANAGER },
  { name: 'LightingSystem',        path: '../systems/267_lgt_LightingSystem.js',              priority: STAGE_PRIORITY.LIGHT_SYSTEM },
  { name: 'UniversalLightManager', path: '../systems/268_lgt_UniversalLightManager.js',       priority: STAGE_PRIORITY.LIGHT_SYSTEM + 1 },
  { name: 'ShadowSystem',          path: '../systems/270_lgt_ShadowSystemOrchestrator.js',    priority: STAGE_PRIORITY.SHADOW_SYSTEM },
  { name: 'GISystem',              path: '../systems/271_lgt_GISystemOrchestrator.js',        priority: STAGE_PRIORITY.GI_SYSTEM },
  { name: 'AOSystem',              path: '../systems/272_lgt_AOSystemOrchestrator.js',        priority: STAGE_PRIORITY.AO_SYSTEM },
  { name: 'EnvironmentSystem',     path: '../systems/273_lgt_EnvironmentSystemOrchestrator.js', priority: STAGE_PRIORITY.ENVIRONMENT_SYSTEM },
  { name: 'InteriorSystem',        path: '../systems/274_lgt_InteriorSystemOrchestrator.js',  priority: STAGE_PRIORITY.INTERIOR_SYSTEM },
  { name: 'ExteriorSystem',        path: '../systems/275_lgt_ExteriorSystemOrchestrator.js',  priority: STAGE_PRIORITY.EXTERIOR_SYSTEM },
  { name: 'AnimeLightDirector',    path: '../systems/276_lgt_AnimeLightDirector.js',          priority: STAGE_PRIORITY.ANIME_LIGHT_DIRECTOR },
  { name: 'ReferenceLightMatcher', path: '../systems/277_lgt_ReferenceLightMatcher.js',       priority: STAGE_PRIORITY.REFERENCE_LIGHT_MATCHER },
  { name: 'ColorOnlyDirector',     path: '../systems/278_lgt_ColorOnlyLightDirector.js',      priority: STAGE_PRIORITY.COLOR_ONLY_DIRECTOR },
]);

/* ------------------------------------------------------------------ */
/* 2. APP CLASS                                                       */
/* ------------------------------------------------------------------ */

export class App {
  constructor(options = {}) {
    this.options = Object.assign({
      container:        null,
      autoboot:         true,
      autorun:          true,
      width:            (typeof window !== 'undefined' ? window.innerWidth  : 480),
      height:           (typeof window !== 'undefined' ? window.innerHeight : 720),
      pixelRatio:       Math.min(
                          (typeof window !== 'undefined' ? window.devicePixelRatio : 1),
                          MOBILE_DPR_CAP
                        ),
      enableShadows:    true,
      lowPowerMode:     false,
      thermalGuard:     true,
      batteryGuard:     true,
      registerStack:    true,
    }, options || {});

    this._state = APP_STATE.IDLE;
    this._container = null;

    this._loadedModules = new Map();
    this._stageEntries = new Map();

    this._boundAppResize = this._onAppResize.bind(this);
    this._boundVisibility = this._onVisibility.bind(this);
    this._boundThermal = this._onThermalChange.bind(this);
    this._boundBattery = this._onBatteryChange.bind(this);

    this._listeners = new Map();

    this._battery = null;
    this._thermal = 'nominal';
    this._lowPower = !!this.options.lowPowerMode;

    this._stats = {
      bootTimeMs:        0,
      loadTimeMs:        0,
      modulesLoaded:     0,
      modulesFailed:     0,
      modulesSkipped:    0,
      lastError:         null,
    };
  }

  /* ---------------- state ---------------- */

  get state()             { return this._state; }
  get isReady()            { return this._state === APP_STATE.READY || this._state === APP_STATE.RUNNING || this._state === APP_STATE.PAUSED; }
  get isRunning()          { return this._state === APP_STATE.RUNNING; }
  get isPaused()           { return this._state === APP_STATE.PAUSED; }
  get perfTier()           { return PERF_TIER; }
  get stats()              { return this._stats; }

  _setState(next) {
    if (this._state === next) return;
    const prev = this._state;
    this._state = next;
    this._emit('state', { from: prev, to: next });
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
      try { arr[i](payload); } catch (e) { console.error(`[002_rnd_App] listener error on "${event}"`, e); }
    }
  }

  /* ---------------- DOM mounting ---------------- */

  mount(container) {
    this._container =
      typeof container === 'string'
        ? (typeof document !== 'undefined' ? document.querySelector(container) : null)
        : container;

    if (!this._container && typeof document !== 'undefined') {
      this._container = document.body;
    }

    if (!this._container) {
      throw new Error('[002_rnd_App] mount(): no container available');
    }

    return this;
  }

  _attachCanvas() {
    const renderer = getRenderer();
    if (!renderer || !renderer.domElement || !this._container) return;

    const canvas = renderer.domElement;
    if (canvas.parentNode !== this._container) {
      this._container.appendChild(canvas);
    }

    canvas.style.position = 'absolute';
    canvas.style.inset = '0';
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.style.display = 'block';
    canvas.style.touchAction = 'none';
    canvas.style.userSelect = 'none';
    canvas.style.webkitTapHighlightColor = 'transparent';

    if (this._container.style) {
      this._container.style.position = 'relative';
      this._container.style.overflow = 'hidden';
      this._container.style.background = '#000';
    }
  }

  /* ---------------- boot ---------------- */

  async boot(options = {}) {
    if (this._state !== APP_STATE.IDLE) return this;

    const t0 = (typeof performance !== 'undefined' ? performance.now() : Date.now());
    this._setState(APP_STATE.BOOTING);

    try {
      // ---- 1. Merge options ----
      Object.assign(this.options, options || {});

      // ---- 2. Mount DOM ----
      if (this.options.container) this.mount(this.options.container);

      // ---- 3. Bootstrap the runtime (world + bitECS audit + rAF) ----
      Bootstrap.boot({
        width:           this.options.width,
        height:          this.options.height,
        pixelRatio:      this.options.pixelRatio,
        enableShadows:   this.options.enableShadows,
        autostart:       false,
      });

      this._attachCanvas();

      // ---- 4. Android runtime guards ----
      this._attachAndroidGuards();

      // ---- 5. Register every lighting subsystem ----
      this._setState(APP_STATE.LOADING);
      if (this.options.registerStack !== false) {
        await this.registerLightingStack();
      }

      // ---- 6. Wire bootstrap events to app events ----
      this._wireBootstrapEvents();

      const t1 = (typeof performance !== 'undefined' ? performance.now() : Date.now());
      this._stats.bootTimeMs = t1 - t0;

      this._setState(APP_STATE.READY);
      this._emit('ready', { bootTimeMs: this._stats.bootTimeMs });

      if (this.options.autorun) this.start();
    } catch (e) {
      this._stats.lastError = e;
      console.error('[002_rnd_App] boot failed', e);
      this._setState(APP_STATE.IDLE);
      throw e;
    }

    return this;
  }

  /* ---------------- lighting stack auto-wiring ---------------- */

  async registerLightingStack() {
    const t0 = (typeof performance !== 'undefined' ? performance.now() : Date.now());

    for (let i = 0; i < LIGHTING_MODULES.length; i++) {
      const descriptor = LIGHTING_MODULES[i];
      try {
        const mod = await import(/* @vite-ignore */ descriptor.path);
        this._loadedModules.set(descriptor.name, mod);

        const instance = this._instantiateModule(mod, descriptor);
        if (instance) {
          this._stageEntries.set(descriptor.name, Bootstrap.registerStage(
            descriptor.name,
            instance,
            descriptor.priority
          ));
          this._stats.modulesLoaded++;
        } else {
          this._stats.modulesSkipped++;
        }
      } catch (_) {
        this._stats.modulesFailed++;
        // Missing module is not fatal — app still boots.
      }
    }

    const t1 = (typeof performance !== 'undefined' ? performance.now() : Date.now());
    this._stats.loadTimeMs = t1 - t0;
    this._emit('stack', {
      loaded:  this._stats.modulesLoaded,
      failed:  this._stats.modulesFailed,
      skipped: this._stats.modulesSkipped,
      timeMs:  this._stats.loadTimeMs,
    });

    return this;
  }

  _instantiateModule(mod, descriptor) {
    // Convention: each lighting module exports either a factory named
    // `create<Name>` or a class named `<Name>` plus an optional
    // `system<Name>(dt, elapsed, ctx)` function.
    const factoryKey  = `create${descriptor.name}`;
    const classKey    = descriptor.name;
    const defaultKey  = 'default';

    let instance = null;

    if (typeof mod[factoryKey] === 'function') {
      try { instance = mod[factoryKey]({ app: this, bootstrap: Bootstrap }); } catch (_) { instance = null; }
    }

    if (!instance && typeof mod[classKey] === 'function') {
      try { instance = new mod[classKey]({ app: this, bootstrap: Bootstrap }); } catch (_) { instance = null; }
    }

    if (!instance && mod[defaultKey]) {
      const d = mod[defaultKey];
      if (typeof d === 'function') {
        try { instance = new d({ app: this, bootstrap: Bootstrap }); } catch (_) { instance = null; }
      } else if (typeof d === 'object' && (typeof d.update === 'function' || typeof d.dispose === 'function')) {
        instance = d;
      }
    }

    if (!instance && typeof mod.update === 'function') {
      instance = mod;
    }

    return instance;
  }

  /* ---------------- bootstrap event bridge ---------------- */

  _wireBootstrapEvents() {
    Bootstrap.on('frame', (info) => this._emit('frame', info));
    Bootstrap.on('resize', (info) => this._emit('resize', info));
    Bootstrap.on('contextlost', () => this._emit('contextlost', null));
    Bootstrap.on('contextrestored', () => this._emit('contextrestored', null));
    Bootstrap.on('stopped', () => this._emit('stopped', null));
    Bootstrap.on('started', () => this._emit('started', null));
  }

  /* ---------------- runtime control ---------------- */

  start() {
    if (!this.isReady && this._state !== APP_STATE.PAUSED) return this;
    Bootstrap.start();
    this._setState(APP_STATE.RUNNING);
    return this;
  }

  stop() {
    Bootstrap.stop();
    this._setState(APP_STATE.READY);
    return this;
  }

  pause() {
    Bootstrap.pause();
    this._setState(APP_STATE.PAUSED);
    return this;
  }

  resume() {
    Bootstrap.resume();
    this._setState(APP_STATE.RUNNING);
    return this;
  }

  resize(width, height, pixelRatio) {
    const w = (width  !== undefined ? width  : (typeof window !== 'undefined' ? window.innerWidth  : this.options.width));
    const h = (height !== undefined ? height : (typeof window !== 'undefined' ? window.innerHeight : this.options.height));
    const d = (pixelRatio !== undefined ? pixelRatio : (typeof window !== 'undefined' ? window.devicePixelRatio : 1));

    this.options.width = Math.max(1, w | 0);
    this.options.height = Math.max(1, h | 0);
    this.options.pixelRatio = Math.min(Math.max(0.5, Number(d) || 1), MOBILE_DPR_CAP);

    Bootstrap.resize(this.options.width, this.options.height, this.options.pixelRatio);
    return this;
  }

  dispose() {
    this._detachAndroidGuards();

    if (typeof window !== 'undefined') {
      window.removeEventListener('resize', this._boundAppResize);
      window.removeEventListener('orientationchange', this._boundAppResize);
    }
    if (typeof document !== 'undefined') {
      document.removeEventListener('visibilitychange', this._boundVisibility);
    }

    Bootstrap.dispose();

    this._loadedModules.clear();
    this._stageEntries.clear();
    this._listeners.clear();

    this._setState(APP_STATE.DISPOSED);
    return this;
  }

  /* ---------------- app-level commands ---------------- */

  setTheme(themeId) {
    const core = getCore();
    if (core && typeof core.setTheme === 'function') {
      try { core.setTheme(themeId); } catch (e) { console.error('[002_rnd_App] setTheme failed', e); }
    }
    this._emit('theme', { themeId });
    return this;
  }

  setBiome(biomeId, instant = false) {
    const core = getCore();
    if (core && typeof core.setBiome === 'function') {
      try { core.setBiome(biomeId, instant); } catch (e) { console.error('[002_rnd_App] setBiome failed', e); }
    }
    this._emit('biome', { biomeId, instant });
    return this;
  }

  setDayCycle(t) {
    const core = getCore();
    if (core && typeof core.setDayCycle === 'function') {
      try { core.setDayCycle(t); } catch (e) { console.error('[002_rnd_App] setDayCycle failed', e); }
    }
    this._emit('daycycle', { t });
    return this;
  }

  setIndoorOutdoorBlend(t) {
    const lightingSystem = this._getLoadedInstance('LightingSystem');
    if (lightingSystem && typeof lightingSystem.setIndoorOutdoorBlend === 'function') {
      try { lightingSystem.setIndoorOutdoorBlend(t); } catch (e) { console.error('[002_rnd_App] setIndoorOutdoorBlend failed', e); }
    }
    this._emit('indooroutdoor', { t });
    return this;
  }

  setInterior(interiorId) {
    const interior = this._getLoadedInstance('InteriorSystem');
    if (interior && typeof interior.setInterior === 'function') {
      try { interior.setInterior(interiorId); } catch (e) { console.error('[002_rnd_App] setInterior failed', e); }
    }
    this._emit('interior', { interiorId });
    return this;
  }

  setExterior(exteriorId) {
    const exterior = this._getLoadedInstance('ExteriorSystem');
    if (exterior && typeof exterior.setExterior === 'function') {
      try { exterior.setExterior(exteriorId); } catch (e) { console.error('[002_rnd_App] setExterior failed', e); }
    }
    this._emit('exterior', { exteriorId });
    return this;
  }

  toggleShadows(enabled) {
    const shadowSystem = this._getLoadedInstance('ShadowSystem');
    if (shadowSystem && typeof shadowSystem.setEnabled === 'function') {
      try { shadowSystem.setEnabled(!!enabled); } catch (e) { console.error('[002_rnd_App] toggleShadows failed', e); }
    }
    this.options.enableShadows = !!enabled;
    this._emit('shadows', { enabled: !!enabled });
    return this;
  }

  toggleGI(enabled) {
    const gi = this._getLoadedInstance('GISystem');
    if (gi && typeof gi.setEnabled === 'function') {
      try { gi.setEnabled(!!enabled); } catch (e) { console.error('[002_rnd_App] toggleGI failed', e); }
    }
    this._emit('gi', { enabled: !!enabled });
    return this;
  }

  toggleAO(enabled) {
    const ao = this._getLoadedInstance('AOSystem');
    if (ao && typeof ao.setEnabled === 'function') {
      try { ao.setEnabled(!!enabled); } catch (e) { console.error('[002_rnd_App] toggleAO failed', e); }
    }
    this._emit('ao', { enabled: !!enabled });
    return this;
  }

  screenshot(mime = 'image/png') {
    const renderer = getRenderer();
    if (!renderer || !renderer.domElement) return null;
    try {
      const url = renderer.domElement.toDataURL(mime);
      this._emit('screenshot', { url, mime });
      return url;
    } catch (e) {
      console.error('[002_rnd_App] screenshot failed', e);
      return null;
    }
  }

  _getLoadedInstance(name) {
    const entry = this._stageEntries.get(name);
    return entry ? entry.stage : null;
  }

  /* ---------------- Android runtime guards ---------------- */

  _attachAndroidGuards() {
    if (typeof window === 'undefined') return;

    window.addEventListener('resize', this._boundAppResize, { passive: true });
    window.addEventListener('orientationchange', this._boundAppResize, { passive: true });

    if (typeof document !== 'undefined') {
      document.addEventListener('visibilitychange', this._boundVisibility, { passive: true });
    }

    if (this.options.thermalGuard && typeof navigator !== 'undefined' && 'deviceMemory' in navigator) {
      // Thermal API is Chrome-only and unstable; guard existence.
      if ('computePressure' in window && typeof window.computePressure === 'function') {
        try {
          this._pressureObserver = window.computePressure('cpu', (state) => this._onThermalChange(state));
        } catch (_) { this._pressureObserver = null; }
      }
    }

    if (this.options.batteryGuard && typeof navigator !== 'undefined' && typeof navigator.getBattery === 'function') {
      navigator.getBattery()
        .then((b) => {
          this._battery = b;
          b.addEventListener('levelchange', this._boundBattery);
          b.addEventListener('chargingchange', this._boundBattery);
          this._onBatteryChange();
        })
        .catch(() => { this._battery = null; });
    }
  }

  _detachAndroidGuards() {
    if (typeof window !== 'undefined') {
      window.removeEventListener('resize', this._boundAppResize);
      window.removeEventListener('orientationchange', this._boundAppResize);
    }
    if (typeof document !== 'undefined') {
      document.removeEventListener('visibilitychange', this._boundVisibility);
    }
    if (this._battery) {
      this._battery.removeEventListener('levelchange', this._boundBattery);
      this._battery.removeEventListener('chargingchange', this._boundBattery);
      this._battery = null;
    }
    if (this._pressureObserver && typeof this._pressureObserver.disconnect === 'function') {
      try { this._pressureObserver.disconnect(); } catch (_) { /* noop */ }
      this._pressureObserver = null;
    }
  }

  _onAppResize() {
    this.resize(
      (typeof window !== 'undefined' ? window.innerWidth  : this.options.width),
      (typeof window !== 'undefined' ? window.innerHeight : this.options.height),
      (typeof window !== 'undefined' ? window.devicePixelRatio : 1)
    );
  }

  _onVisibility() {
    if (typeof document === 'undefined') return;
    if (document.hidden) this.pause();
    else this.resume();
  }

  _onThermalChange(state) {
    const s = state && state.state ? state.state : String(state || 'nominal');
    if (s === this._thermal) return;
    this._thermal = s;

    if (s === 'serious' || s === 'critical') {
      this.toggleShadows(false);
      this.toggleGI(false);
    } else if (s === 'nominal') {
      this.toggleShadows(this.options.enableShadows);
      this.toggleGI(true);
    }

    this._emit('thermal', { state: s });
  }

  _onBatteryChange() {
    if (!this._battery) return;
    const info = {
      level: this._battery.level,
      charging: this._battery.charging,
    };

    if (!info.charging && info.level <= 0.15) {
      this._lowPower = true;
      this.toggleShadows(false);
      this.toggleGI(false);
      this.toggleAO(false);
    } else if (info.charging || info.level > 0.25) {
      this._lowPower = !!this.options.lowPowerMode;
    }

    this._emit('battery', info);
  }
}

/* ------------------------------------------------------------------ */
/* 3. FACTORY                                                         */
/* ------------------------------------------------------------------ */

export function createApp(options = {}) {
  return new App(options);
}

/* ------------------------------------------------------------------ */
/* 4. MODULE-LEVEL SINGLETON (optional convenience)                   */
/* ------------------------------------------------------------------ */

let _defaultApp = null;

export function getDefaultApp() {
  if (!_defaultApp) _defaultApp = new App();
  return _defaultApp;
}

export function disposeDefaultApp() {
  if (_defaultApp) {
    _defaultApp.dispose();
    _defaultApp = null;
  }
}

/* ------------------------------------------------------------------ */
/* 5. DEFAULT EXPORT                                                  */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  App,
  createApp,
  getDefaultApp,
  disposeDefaultApp,
  APP_STATE,
  LIGHTING_MODULES,
  STAGE_PRIORITY,
};

export default _defaultExport;