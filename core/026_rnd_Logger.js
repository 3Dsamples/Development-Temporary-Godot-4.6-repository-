// File : 026
// name : src/core/026_rnd_Logger.js
// description : Structured logging and diagnostics hub for the anime lighting
//               stack on Android mobile. Owns every diagnostic message the
//               engine emits: startup banners, module load confirmations,
//               shader compile successes/failures, pool exhaustion warnings,
//               registry leak reports, tier transitions, quality downgrades,
//               hitch warnings, and runtime errors.
//
//               Design:
//                 • Level-gated: TRACE / DEBUG / INFO / WARN / ERROR / FATAL
//                   with a per-channel override map so the lighting stack
//                   can be loud about shadows but silent about geometry.
//                 • Channel-based: every emitter passes a channel id
//                   (CORE / LIGHTS / SHADOWS / GI / AO / CLUSTER / ENV /
//                   INTERIOR / EXTERIOR / POST / POOL / REGISTRY / MANIFEST
//                   / PLATFORM / PROFILE / SCHEDULER / WORKER / GPU). The
//                   channel override map uses typed-array slots, O(1).
//                 • Zero-alloc hot path: when a message's level is below the
//                   effective threshold for its channel, the call returns
//                   immediately with one integer compare. No template
//                   strings, no array pushes, no closures.
//                 • Lazy evaluation: callers pass either a plain string or a
//                   thunk `() => string` so expensive messages only build
//                   their string when they will actually be emitted.
//                 • Ring-buffer history: the last N log entries are kept in
//                   a fixed-size ring so a debug HUD or regression tool can
//                   pull the tail without re-parsing the console.
//                 • Console output: gated per level (INFO never spams the
//                   console by default, WARN/ERROR always do). Uses
//                   console.log/warn/error directly, no wrapper objects.
//                 • Sink registry: named sinks receive every emitted entry
//                   in addition to the ring buffer + console. A sink can
//                   forward to a debug HUD, write to a remote endpoint, or
//                   count errors by channel. Zero-alloc sinks are the norm.
//                 • Android-specific: `navigator.onLine` gate for remote
//                   sinks, `navigator.connection.saveData` gate for
//                   verbose modes, `performance.now()` for high-res
//                   timestamps, monotonic frame counter for the hot path.
//                 • Integrates with 024_rnd_Profiler.js: WARN/ERROR entries
//                   can optionally mark the current frame in the profiler
//                   so a hitch correlates with the log line that caused it.
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0 API
//               only; no external logging libs; every internal array sized
//               once at construction.
// best for : Giving every subsystem in the anime lighting stack one
//            consistent, greppable, level-controlled diagnostic channel.
//            Debug HUD, CI, screenshot regression, and crash reporting all
//            read from the same ring buffer with the same channel ids.
// License : GPL-3.0
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import {
  getPerfTier,
} from './008_scn_world.js';

import {
  getDefaultProfiler,
  profilerMark,
} from './024_rnd_Profiler.js';

import {
  PLATFORM,
  DEVICE,
} from './018_rnd_PlatformConfig.js';

import {
  getNetworkInfo,
} from './022_rnd_FeatureDetector.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER_LOCAL = getPerfTier();

export const LOG_LEVEL = Object.freeze({
  TRACE: 0,
  DEBUG: 1,
  INFO:  2,
  WARN:  3,
  ERROR: 4,
  FATAL: 5,
  OFF:   6,
  COUNT: 7,
});

export const LOG_LEVEL_NAME = Object.freeze([
  'TRACE',
  'DEBUG',
  'INFO',
  'WARN',
  'ERROR',
  'FATAL',
  'OFF',
]);

export const LOG_CHANNEL = Object.freeze({
  CORE:      0,
  LIGHTS:    1,
  SHADOWS:   2,
  GI:        3,
  AO:        4,
  CLUSTER:   5,
  ENV:       6,
  INTERIOR:  7,
  EXTERIOR:  8,
  POST:      9,
  POOL:     10,
  REGISTRY: 11,
  MANIFEST: 12,
  PLATFORM: 13,
  PROFILE:  14,
  SCHEDULER:15,
  WORKER:   16,
  GPU:      17,
  ASSET:    18,
  CONFIG:   19,
  QUALITY:  20,
  PERF:     21,
  COUNT:    22,
});

export const LOG_CHANNEL_NAME = Object.freeze([
  'CORE',
  'LIGHTS',
  'SHADOWS',
  'GI',
  'AO',
  'CLUSTER',
  'ENV',
  'INTERIOR',
  'EXTERIOR',
  'POST',
  'POOL',
  'REGISTRY',
  'MANIFEST',
  'PLATFORM',
  'PROFILE',
  'SCHEDULER',
  'WORKER',
  'GPU',
  'ASSET',
  'CONFIG',
  'QUALITY',
  'PERF',
]);

const DEFAULT_HISTORY_CAPACITY =
  PERF_TIER_LOCAL === 'HIGH'   ? 512 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 256 :
                                 128;

const DEFAULT_GLOBAL_LEVEL =
  PERF_TIER_LOCAL === 'HIGH'   ? LOG_LEVEL.DEBUG :
  PERF_TIER_LOCAL === 'MEDIUM' ? LOG_LEVEL.INFO :
                                 LOG_LEVEL.WARN;

const MAX_SINKS = 16;

/* ------------------------------------------------------------------ */
/* 1. HELPERS                                                         */
/* ------------------------------------------------------------------ */

function _now() {
  return (typeof performance !== 'undefined' && typeof performance.now === 'function')
    ? performance.now()
    : Date.now();
}

function _padLeft(s, n, c) {
  s = String(s);
  while (s.length < n) s = c + s;
  return s;
}

function _formatMs(ms) {
  return ms.toFixed(2).padStart(8, ' ') + 'ms';
}

function _formatFrame(f) {
  return '#' + _padLeft(f, 6, ' ');
}

/* ------------------------------------------------------------------ */
/* 2. LOG ENTRY (fixed, reused ring buffer slot)                      */
/* ------------------------------------------------------------------ */

export class LogEntry {
  constructor() {
    this.frame     = 0;
    this.seq       = 0;
    this.timeMs    = 0;
    this.level     = LOG_LEVEL.INFO;
    this.channel   = LOG_CHANNEL.CORE;
    this.message   = '';
    this.data      = null;
  }

  reset() {
    this.frame = 0;
    this.seq = 0;
    this.timeMs = 0;
    this.level = LOG_LEVEL.INFO;
    this.channel = LOG_CHANNEL.CORE;
    this.message = '';
    this.data = null;
    return this;
  }
}

/* ------------------------------------------------------------------ */
/* 3. LOGGER                                                          */
/* ------------------------------------------------------------------ */

export class Logger {
  constructor(options = {}) {
    this.options = Object.assign({
      globalLevel:     DEFAULT_GLOBAL_LEVEL,
      historyCapacity: DEFAULT_HISTORY_CAPACITY,
      echoToConsole:   true,
      consoleLevel:    LOG_LEVEL.INFO,
      stampFrame:      true,
      stampTime:       true,
      throwOnFatal:    false,
      correlateWithProfiler: true,
    }, options || {});

    // Per-level/per-channel gate. `levelGate[channel]` is the minimum
    // level that will be emitted for that channel. `-1` = use global.
    this.globalLevel = this.options.globalLevel;
    this.levelGate   = new Int8Array(LOG_CHANNEL.COUNT).fill(-1);
    this.enabled     = new Uint8Array(LOG_CHANNEL.COUNT).fill(1);

    // History ring.
    this.historyCapacity = this.options.historyCapacity;
    this.history = new Array(this.historyCapacity);
    for (let i = 0; i < this.historyCapacity; i++) this.history[i] = new LogEntry();
    this.historyHead = 0;
    this.historyCount = 0;

    // Per-channel stats (used by debug HUD).
    this.channelCounts = new Uint32Array(LOG_CHANNEL.COUNT * LOG_LEVEL.COUNT);
    this.totalEmitted = 0;
    this.totalSuppressed = 0;

    // Sinks.
    this.sinks = new Array(MAX_SINKS);
    for (let i = 0; i < MAX_SINKS; i++) this.sinks[i] = null;
    this.sinkCount = 0;

    // Sequence and frame tracking.
    this.sequence = 0;
    this.frame = 0;

    // Network state (used to gate remote sinks).
    this._networkInfo = null;

    // Pre-allocated per-emit buffer for the message (reused for the ring).
    this._msgBuf = null;

    // Startup banner (only in HIGH tier).
    if (PERF_TIER_LOCAL === 'HIGH') {
      this._emitStartupBanner();
    }
  }

  /* ---------------- per-frame tick ---------------- */

  beginFrame(frameNumber) {
    this.frame = (typeof frameNumber === 'number') ? frameNumber : (this.frame + 1);
  }

  /* ---------------- gate control ---------------- */

  setGlobalLevel(level) {
    if (level < 0 || level >= LOG_LEVEL.COUNT) return false;
    this.globalLevel = level | 0;
    return true;
  }

  getGlobalLevel() { return this.globalLevel; }

  setChannelLevel(channel, level) {
    if (channel < 0 || channel >= LOG_CHANNEL.COUNT) return false;
    if (level === null || level === undefined) {
      this.levelGate[channel] = -1;
      return true;
    }
    if (level < 0 || level >= LOG_LEVEL.COUNT) return false;
    this.levelGate[channel] = level | 0;
    return true;
  }

  getChannelLevel(channel) {
    if (channel < 0 || channel >= LOG_CHANNEL.COUNT) return this.globalLevel;
    const v = this.levelGate[channel];
    return v < 0 ? this.globalLevel : v;
  }

  setChannelEnabled(channel, enabled) {
    if (channel < 0 || channel >= LOG_CHANNEL.COUNT) return false;
    this.enabled[channel] = enabled ? 1 : 0;
    return true;
  }

  isChannelEnabled(channel) {
    if (channel < 0 || channel >= LOG_CHANNEL.COUNT) return false;
    return this.enabled[channel] === 1;
  }

  /* ---------------- emission ---------------- */

  isEnabled(level, channel) {
    if (level < 0 || level >= LOG_LEVEL.COUNT) return false;
    if (channel < 0 || channel >= LOG_CHANNEL.COUNT) return false;
    if (this.enabled[channel] === 0) return false;
    const gate = this.levelGate[channel];
    const minLevel = (gate < 0) ? this.globalLevel : gate;
    if (minLevel >= LOG_LEVEL.OFF) return false;
    return level >= minLevel;
  }

  /**
   * Main emit. `messageOrThunk` may be a string or a function returning a
   * string. When the level is gated off, the thunk is never called — that
   * is the whole point of accepting a thunk.
   */
  emit(level, channel, messageOrThunk, data) {
    if (!this.isEnabled(level, channel)) {
      this.totalSuppressed++;
      return false;
    }

    let message;
    if (typeof messageOrThunk === 'function') {
      try { message = messageOrThunk(); }
      catch (e) { message = '<log thunk threw: ' + (e && e.message) + '>'; }
    } else {
      message = String(messageOrThunk);
    }

    // Record into ring.
    const entry = this.history[this.historyHead];
    entry.reset();
    entry.frame   = this.frame;
    entry.seq     = this.sequence++;
    entry.timeMs  = _now();
    entry.level   = level;
    entry.channel = channel;
    entry.message = message;
    entry.data    = data || null;

    this.historyHead = (this.historyHead + 1) % this.historyCapacity;
    if (this.historyCount < this.historyCapacity) this.historyCount++;

    // Per-channel stats.
    this.channelCounts[channel * LOG_LEVEL.COUNT + level]++;
    this.totalEmitted++;

    // Console echo.
    if (this.options.echoToConsole && level >= this.options.consoleLevel) {
      this._writeConsole(entry);
    }

    // Sinks.
    for (let i = 0; i < this.sinkCount; i++) {
      const sink = this.sinks[i];
      if (!sink) continue;
      try { sink(entry); }
      catch (e) { /* swallow sink errors */ }
    }

    // Profiler correlation (only for WARN and above, and only when enabled).
    if (this.options.correlateWithProfiler && level >= LOG_LEVEL.WARN) {
      try {
        const p = getDefaultProfiler();
        p.mark('log:' + LOG_CHANNEL_NAME[channel] + ':' + LOG_LEVEL_NAME[level]);
      } catch (_) { /* swallow */ }
    }

    // Fatal.
    if (level === LOG_LEVEL.FATAL && this.options.throwOnFatal) {
      throw new Error('[Logger] FATAL: ' + message);
    }

    return true;
  }

  /* ---------------- level helpers ---------------- */

  trace(channel, messageOrThunk, data) {
    return this.emit(LOG_LEVEL.TRACE, channel, messageOrThunk, data);
  }

  debug(channel, messageOrThunk, data) {
    return this.emit(LOG_LEVEL.DEBUG, channel, messageOrThunk, data);
  }

  info(channel, messageOrThunk, data) {
    return this.emit(LOG_LEVEL.INFO, channel, messageOrThunk, data);
  }

  warn(channel, messageOrThunk, data) {
    return this.emit(LOG_LEVEL.WARN, channel, messageOrThunk, data);
  }

  error(channel, messageOrThunk, data) {
    return this.emit(LOG_LEVEL.ERROR, channel, messageOrThunk, data);
  }

  fatal(channel, messageOrThunk, data) {
    return this.emit(LOG_LEVEL.FATAL, channel, messageOrThunk, data);
  }

  /* ---------------- console output ---------------- */

  _writeConsole(entry) {
    const ts = this.options.stampTime
      ? '[' + entry.timeMs.toFixed(0).padStart(7, ' ') + 'ms]'
      : '';
    const fr = this.options.stampFrame
      ? _formatFrame(entry.frame)
      : '';
    const lvl = LOG_LEVEL_NAME[entry.level].padEnd(5, ' ');
    const chn = LOG_CHANNEL_NAME[entry.channel].padEnd(9, ' ');
    const line = `${ts} ${fr} ${lvl} [${chn}] ${entry.message}`;

    switch (entry.level) {
      case LOG_LEVEL.TRACE:
      case LOG_LEVEL.DEBUG:
        if (console.debug) console.debug(line); else console.log(line);
        break;
      case LOG_LEVEL.INFO:
        console.log(line);
        break;
      case LOG_LEVEL.WARN:
        console.warn(line);
        break;
      case LOG_LEVEL.ERROR:
      case LOG_LEVEL.FATAL:
        console.error(line);
        break;
      default:
        console.log(line);
    }
  }

  /* ---------------- sinks ---------------- */

  addSink(fn) {
    if (typeof fn !== 'function') return false;
    if (this.sinkCount >= MAX_SINKS) return false;
    this.sinks[this.sinkCount++] = fn;
    return true;
  }

  removeSink(fn) {
    for (let i = 0; i < this.sinkCount; i++) {
      if (this.sinks[i] === fn) {
        for (let j = i; j < this.sinkCount - 1; j++) {
          this.sinks[j] = this.sinks[j + 1];
        }
        this.sinks[this.sinkCount - 1] = null;
        this.sinkCount--;
        return true;
      }
    }
    return false;
  }

  /* ---------------- history ---------------- */

  /**
   * Copies up to `max` recent entries into a caller-provided array.
   * Returns the number copied. The caller owns the array.
   */
  copyHistory(max, out) {
    if (!Array.isArray(out)) return 0;
    const n = Math.min(max, this.historyCount);
    for (let i = 0; i < n; i++) {
      const idx = (this.historyHead - n + i + this.historyCapacity) % this.historyCapacity;
      out[i] = this.history[idx];
    }
    return n;
  }

  getRecentEntry(offsetFromHead) {
    const k = offsetFromHead | 0;
    if (k < 0 || k >= this.historyCount) return null;
    const idx = (this.historyHead - 1 - k + this.historyCapacity) % this.historyCapacity;
    return this.history[idx];
  }

  /**
   * Formats the last `max` entries as a plain string (for copy/paste or
   * regression dumps). Allocates a fresh string — use off the hot path.
   */
  dump(max) {
    const n = Math.min(max | 0 || this.historyCount, this.historyCount);
    let out = '';
    for (let i = 0; i < n; i++) {
      const idx = (this.historyHead - n + i + this.historyCapacity) % this.historyCapacity;
      const e = this.history[idx];
      out += `${_formatMs(e.timeMs)} ${_formatFrame(e.frame)} ${LOG_LEVEL_NAME[e.level].padEnd(5, ' ')} [${LOG_CHANNEL_NAME[e.channel]}] ${e.message}\n`;
    }
    return out;
  }

  /**
   * Returns a summary of per-channel emissions.
   */
  getChannelSummary() {
    const out = new Array(LOG_CHANNEL.COUNT);
    for (let ch = 0; ch < LOG_CHANNEL.COUNT; ch++) {
      const counts = new Array(LOG_LEVEL.COUNT);
      let total = 0;
      for (let lvl = 0; lvl < LOG_LEVEL.COUNT; lvl++) {
        const c = this.channelCounts[ch * LOG_LEVEL.COUNT + lvl];
        counts[lvl] = c;
        total += c;
      }
      out[ch] = {
        channel: LOG_CHANNEL_NAME[ch],
        total,
        counts,
      };
    }
    return out;
  }

  /* ---------------- diagnostics ---------------- */

  getStats() {
    return {
      globalLevel:      LOG_LEVEL_NAME[this.globalLevel],
      globalLevelIndex: this.globalLevel,
      historyCapacity:  this.historyCapacity,
      historyCount:     this.historyCount,
      totalEmitted:     this.totalEmitted,
      totalSuppressed:  this.totalSuppressed,
      sinkCount:        this.sinkCount,
      channelSummary:   this.getChannelSummary(),
    };
  }

  /* ---------------- startup banner ---------------- */

  _emitStartupBanner() {
    const isAndroid = DEVICE.isAndroid;
    const isMobile  = DEVICE.isMobile;
    const tier      = PERF_TIER_LOCAL;

    this.info(LOG_CHANNEL.CORE, () =>
      `astro_loop anime lighting stack — boot (tier=${tier}, ` +
      `platform=${isAndroid ? 'android' : isMobile ? 'mobile' : 'desktop'})`
    );
  }

  /* ---------------- reset / dispose ---------------- */

  reset() {
    for (let i = 0; i < this.historyCapacity; i++) this.history[i].reset();
    this.historyHead = 0;
    this.historyCount = 0;
    this.channelCounts.fill(0);
    this.totalEmitted = 0;
    this.totalSuppressed = 0;
    this.sequence = 0;
    this.frame = 0;
    return this;
  }

  dispose() {
    this.reset();
    for (let i = 0; i < MAX_SINKS; i++) this.sinks[i] = null;
    this.sinkCount = 0;
    this.history.length = 0;
    this.history = null;
    return this;
  }
}

/* ------------------------------------------------------------------ */
/* 4. MODULE-LEVEL SINGLETON                                          */
/* ------------------------------------------------------------------ */

let _defaultLogger = null;

export function getDefaultLogger() {
  if (!_defaultLogger) _defaultLogger = new Logger();
  return _defaultLogger;
}

export function disposeDefaultLogger() {
  if (_defaultLogger) {
    _defaultLogger.dispose();
    _defaultLogger = null;
  }
}

/* ------------------------------------------------------------------ */
/* 5. HOT-PATH HELPERS (delegate to default logger)                   */
/* ------------------------------------------------------------------ */

export function logBeginFrame(frameNumber) {
  getDefaultLogger().beginFrame(frameNumber);
}

export function logTrace(channel, messageOrThunk, data) {
  return getDefaultLogger().trace(channel, messageOrThunk, data);
}

export function logDebug(channel, messageOrThunk, data) {
  return getDefaultLogger().debug(channel, messageOrThunk, data);
}

export function logInfo(channel, messageOrThunk, data) {
  return getDefaultLogger().info(channel, messageOrThunk, data);
}

export function logWarn(channel, messageOrThunk, data) {
  return getDefaultLogger().warn(channel, messageOrThunk, data);
}

export function logError(channel, messageOrThunk, data) {
  return getDefaultLogger().error(channel, messageOrThunk, data);
}

export function logFatal(channel, messageOrThunk, data) {
  return getDefaultLogger().fatal(channel, messageOrThunk, data);
}

export function logSetGlobalLevel(level) {
  return getDefaultLogger().setGlobalLevel(level);
}

export function logSetChannelLevel(channel, level) {
  return getDefaultLogger().setChannelLevel(channel, level);
}

export function logSetChannelEnabled(channel, enabled) {
  return getDefaultLogger().setChannelEnabled(channel, enabled);
}

export function logDump(max) {
  return getDefaultLogger().dump(max);
}

/* ------------------------------------------------------------------ */
/* 6. CONVENIENCE SHORTCUTS (channel-bound emitters)                  */
/* ------------------------------------------------------------------ */

export function createChannelLogger(channel) {
  const logger = getDefaultLogger();
  return {
    channel,
    trace: (m, d) => logger.trace(channel, m, d),
    debug: (m, d) => logger.debug(channel, m, d),
    info:  (m, d) => logger.info(channel, m, d),
    warn:  (m, d) => logger.warn(channel, m, d),
    error: (m, d) => logger.error(channel, m, d),
    fatal: (m, d) => logger.fatal(channel, m, d),
  };
}

/* ------------------------------------------------------------------ */
/* 7. FACTORY                                                         */
/* ------------------------------------------------------------------ */

export function createLogger(options = {}) {
  return new Logger(options);
}

/* ------------------------------------------------------------------ */
/* 8. DEFAULT EXPORT                                                  */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  Logger,
  LogEntry,

  createLogger,
  getDefaultLogger,
  disposeDefaultLogger,

  logBeginFrame,
  logTrace,
  logDebug,
  logInfo,
  logWarn,
  logError,
  logFatal,
  logSetGlobalLevel,
  logSetChannelLevel,
  logSetChannelEnabled,
  logDump,

  createChannelLogger,

  LOG_LEVEL,
  LOG_LEVEL_NAME,
  LOG_CHANNEL,
  LOG_CHANNEL_NAME,
  DEFAULT_HISTORY_CAPACITY,
  DEFAULT_GLOBAL_LEVEL,
  MAX_SINKS,
};

export default _defaultExport;