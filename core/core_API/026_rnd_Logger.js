API Documentation — src/core/026_rnd_Logger.js

File Purpose

This file is the structured logging and diagnostics hub for the anime lighting stack. It owns every diagnostic message the engine emits: startup banners, module load confirmations, shader compile successes and failures, pool exhaustion warnings, registry leak reports, tier transitions, quality downgrades, hitch warnings, and runtime errors.

The problem it solves is that a complex engine with forty-plus subsystems needs a way to emit diagnostics that is:

1. Level-gated, so debug output does not spam the console in production.
2. Channel-organized, so a message about shadow atlas packing is distinguishable from a message about GI probe baking.
3. Cheap on the hot path, so a subsystem can emit a debug message without paying for the string formatting if the level is disabled.
4. Persistent, so the last N messages can be pulled from a debug HUD or a crash reporter without re-parsing the console.
5. Forwardable, so an external sink can capture every message for a regression run or a remote logging endpoint.

The logger achieves all five with a fixed structure:

· Levels: TRACE, DEBUG, INFO, WARN, ERROR, FATAL, plus OFF for disabling entirely.
· Channels: twenty-two named categories (CORE, LIGHTS, SHADOWS, GI, AO, CLUSTER, ENV, INTERIOR, EXTERIOR, POST, POOL, REGISTRY, MANIFEST, PLATFORM, PROFILE, SCHEDULER, WORKER, GPU, ASSET, CONFIG, QUALITY, PERF).
· Level gates: a global minimum level plus per-channel overrides so a subsystem can be noisy about its own diagnostics and silent about everything else.
· Thunks: a message may be a string or a function returning a string. When the level is gated off, the thunk is never called, so expensive string construction is skipped entirely.
· History ring: the last N entries are kept in a fixed-size ring buffer. A debug HUD can pull the tail; a crash handler can dump it.
· Sinks: named callbacks that receive every emitted entry in addition to the ring and console. Zero allocations on the hot path.

The logger also integrates with the profiler: when a WARN or higher is emitted, it records a marker in the profiler so a hitch or a frame drop can be correlated with the log line that caused it. That correlation is what makes the logger useful for debugging performance issues, not just for reading diagnostics.

---

Exported Constants

PERF_TIER_LOCAL

Internal. The cached PERF_TIER string from getPerfTier().

LOG_LEVEL

Type: frozen enum

Values:

· TRACE = 0
· DEBUG = 1
· INFO = 2
· WARN = 3
· ERROR = 4
· FATAL = 5
· OFF = 6
· COUNT = 7

Lower integer values mean more verbose. A message is emitted if its level is at least the effective minimum level for its channel.

LOG_LEVEL_NAME

Type: frozen array

Values: ['TRACE', 'DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL', 'OFF'].

LOG_CHANNEL

Type: frozen enum

Values:

· CORE = 0
· LIGHTS = 1
· SHADOWS = 2
· GI = 3
· AO = 4
· CLUSTER = 5
· ENV = 6
· INTERIOR = 7
· EXTERIOR = 8
· POST = 9
· POOL = 10
· REGISTRY = 11
· MANIFEST = 12
· PLATFORM = 13
· PROFILE = 14
· SCHEDULER = 15
· WORKER = 16
· GPU = 17
· ASSET = 18
· CONFIG = 19
· QUALITY = 20
· PERF = 21
· COUNT = 22

LOG_CHANNEL_NAME

Type: frozen array

Maps the enum values back to strings: ['CORE', 'LIGHTS', 'SHADOWS', 'GI', 'AO', 'CLUSTER', 'ENV', 'INTERIOR', 'EXTERIOR', 'POST', 'POOL', 'REGISTRY', 'MANIFEST', 'PLATFORM', 'PROFILE', 'SCHEDULER', 'WORKER', 'GPU', 'ASSET', 'CONFIG', 'QUALITY', 'PERF'].

DEFAULT_HISTORY_CAPACITY

Type: number

Value: 512 on HIGH, 256 on MEDIUM, 128 on LOW.

The default ring buffer size for log history.

DEFAULT_GLOBAL_LEVEL

Type: number

Value: LOG_LEVEL.DEBUG on HIGH, LOG_LEVEL.INFO on MEDIUM, LOG_LEVEL.WARN on LOW.

The default minimum level. On a flagship device the developer sees debug output; on a low-tier device only warnings and errors.

MAX_SINKS

Type: number

Value: 16

The maximum number of registered sink callbacks.

---

Module-Level State (Not Exported Directly)

_defaultLogger

Type: Logger | null

The module-level singleton.

---

Internal Helper Functions (Documented)

_now()

Returns: the current high-resolution timestamp.

_padLeft(s, n, c)

Parameters:

· s — the value to pad.
· n — the target width.
· c — the padding character.

Returns: the padded string. Used to align the console output.

_formatMs(ms)

Parameters: ms — milliseconds.

Returns: a fixed-width string like "  123.45ms".

_formatFrame(f)

Parameters: f — the frame number.

Returns: a fixed-width string like "#   123".

---

Exported Class — LogEntry

One instance per emitted message.

Constructor

```
new LogEntry()
```

Instance Properties

· frame — the frame number at emission.
· seq — a monotonic sequence number.
· timeMs — the high-resolution timestamp.
· level — one of LOG_LEVEL.
· channel — one of LOG_CHANNEL.
· message — the formatted message string.
· data — an optional attached data payload.

Instance Methods

reset()

Returns: this. Zeroes every field.

---

Exported Class — Logger

The main logger.

Constructor

```
new Logger(options = {})
```

Parameters:

· globalLevel — the default minimum level. Default DEFAULT_GLOBAL_LEVEL.
· historyCapacity — the ring buffer size. Default DEFAULT_HISTORY_CAPACITY.
· echoToConsole — whether to write to the console. Default true.
· consoleLevel — the minimum level at which console output occurs. Default LOG_LEVEL.INFO.
· stampFrame — whether to prefix console output with the frame number. Default true.
· stampTime — whether to prefix console output with the timestamp. Default true.
· throwOnFatal — if true, a FATAL message throws instead of just logging. Default false.
· correlateWithProfiler — if true, a WARN or higher records a profiler marker. Default true.

Constructor work:

1. Stores options.
2. Initializes globalLevel.
3. Allocates levelGate — an Int8Array(LOG_CHANNEL.COUNT) initialized to -1 (meaning "use global").
4. Allocates enabled — a Uint8Array(LOG_CHANNEL.COUNT) initialized to 1.
5. Allocates history — an array of LogEntry instances.
6. Initializes the history ring pointers.
7. Allocates channelCounts — a Uint32Array(LOG_CHANNEL.COUNT * LOG_LEVEL.COUNT) for per-channel-per-level counters.
8. Initializes totalEmitted and totalSuppressed.
9. Allocates sinks — an array of 16 entries.
10. Initializes sequence and frame.
11. On HIGH tier, emits a startup banner.

Instance Properties

· options — the merged options.
· globalLevel — the current default minimum level.
· levelGate — the per-channel level overrides.
· enabled — the per-channel enable flags.
· historyCapacity — the ring size.
· history — the ring of LogEntry objects.
· historyHead, historyCount — the ring pointers.
· channelCounts — the per-channel-per-level counters.
· totalEmitted, totalSuppressed — the aggregate counters.
· sinks, sinkCount — the sink registry.
· sequence, frame — the monotonic counters.

Instance Methods

beginFrame(frameNumber)

Parameters: frameNumber — the current frame number.

Returns: nothing.

Purpose: sets this.frame. Called once per frame by the EngineLoop.

setGlobalLevel(level)

Parameters: level — one of LOG_LEVEL.

Returns: boolean.

Purpose: sets the default minimum level.

getGlobalLevel()

Returns: the current global minimum level.

setChannelLevel(channel, level)

Parameters:

· channel — one of LOG_CHANNEL.
· level — one of LOG_LEVEL, or null to fall back to the global level.

Returns: boolean.

Purpose: sets a per-channel override. When set to null, the channel uses the global level.

getChannelLevel(channel)

Parameters: channel — one of LOG_CHANNEL.

Returns: the effective minimum level for the channel.

setChannelEnabled(channel, enabled)

Parameters:

· channel — one of LOG_CHANNEL.
· enabled — boolean.

Returns: boolean.

Purpose: completely disables or re-enables a channel.

isChannelEnabled(channel)

Parameters: channel — one of LOG_CHANNEL.

Returns: boolean.

isEnabled(level, channel)

Parameters:

· level — one of LOG_LEVEL.
· channel — one of LOG_CHANNEL.

Returns: boolean.

Purpose: the fast gate check. Single integer compares. Called by every emit.

emit(level, channel, messageOrThunk, data)

Parameters:

· level — one of LOG_LEVEL.
· channel — one of LOG_CHANNEL.
· messageOrThunk — a string or a function returning a string.
· data — an optional attached data payload.

Returns: boolean — true if the message was emitted.

Purpose: the core emission routine.

Flow:

1. Checks isEnabled. If false, increments totalSuppressed and returns false immediately, without calling the thunk.
2. If messageOrThunk is a function, calls it in try/catch. Otherwise coerces it to a string.
3. Writes the entry into the ring buffer.
4. Updates the per-channel-per-level counters and increments totalEmitted.
5. If echoToConsole and the level is at least consoleLevel, calls _writeConsole.
6. Iterates every registered sink and calls it in try/catch.
7. If correlateWithProfiler and the level is at least WARN, records a profiler marker.
8. If throwOnFatal and the level is FATAL, throws.

trace(channel, messageOrThunk, data)

Convenience wrapper for emit(LOG_LEVEL.TRACE, ...).

debug(channel, messageOrThunk, data)

Convenience wrapper for emit(LOG_LEVEL.DEBUG, ...).

info(channel, messageOrThunk, data)

Convenience wrapper for emit(LOG_LEVEL.INFO, ...).

warn(channel, messageOrThunk, data)

Convenience wrapper for emit(LOG_LEVEL.WARN, ...).

error(channel, messageOrThunk, data)

Convenience wrapper for emit(LOG_LEVEL.ERROR, ...).

fatal(channel, messageOrThunk, data)

Convenience wrapper for emit(LOG_LEVEL.FATAL, ...).

_writeConsole(entry)

Internal. Formats the entry as a single line with optional timestamp, optional frame number, level tag, channel tag, and message. Dispatches to console.log, console.warn, console.error, or console.debug based on the level.

addSink(fn)

Parameters: fn — a callback (entry) => void.

Returns: boolean.

Purpose: registers a sink that receives every emitted entry.

removeSink(fn)

Parameters: fn — the callback to remove.

Returns: boolean.

copyHistory(max, out)

Parameters:

· max — the maximum number of entries to copy.
· out — an array to receive them.

Returns: the number of entries copied.

Purpose: copies the most recent entries into a caller-owned array in chronological order.

getRecentEntry(offsetFromHead)

Parameters: offsetFromHead — how far back from the newest entry to read.

Returns: the LogEntry, or null.

Purpose: reads a single entry from the ring by reverse index.

dump(max)

Parameters: max — the maximum number of entries to include.

Returns: a plain string with one line per entry.

Purpose: allocates a fresh string. Use for copy/paste or regression dumps, not on the hot path.

getChannelSummary()

Returns: an array of per-channel summaries with channel, total, and counts (an array of per-level counts).

getStats()

Returns: an object with globalLevel, globalLevelIndex, historyCapacity, historyCount, totalEmitted, totalSuppressed, sinkCount, channelSummary.

Purpose: the human-readable summary for the debug HUD.

_emitStartupBanner()

Internal. Emits a single INFO-level CORE message with the tier and platform.

reset()

Returns: this. Zeroes every counter and clears the history.

dispose()

Returns: nothing. Resets, clears the sinks, and nulls the history.

---

Exported Hot-Path Wrapper Functions

These delegate to the module-level singleton.

· logBeginFrame(frameNumber)
· logTrace(channel, messageOrThunk, data)
· logDebug(channel, messageOrThunk, data)
· logInfo(channel, messageOrThunk, data)
· logWarn(channel, messageOrThunk, data)
· logError(channel, messageOrThunk, data)
· logFatal(channel, messageOrThunk, data)
· logSetGlobalLevel(level)
· logSetChannelLevel(channel, level)
· logSetChannelEnabled(channel, enabled)
· logDump(max)

---

Exported Function — createChannelLogger(channel)

Parameters: channel — one of LOG_CHANNEL.

Returns: an object with channel, trace, debug, info, warn, error, and fatal methods, all pre-bound to the channel. Downstream modules that emit many messages can create a channel logger once and call its methods without passing the channel every time.

---

Exported Functions

getDefaultLogger()

Returns: the module-level singleton Logger, creating it on first call.

disposeDefaultLogger()

Returns: nothing.

createLogger(options = {})

Returns: a new Logger.

---

Default Export

The default export bundles: Logger, LogEntry, createLogger, getDefaultLogger, disposeDefaultLogger, the eleven log* hot-path wrappers, createChannelLogger, LOG_LEVEL, LOG_LEVEL_NAME, LOG_CHANNEL, LOG_CHANNEL_NAME, DEFAULT_HISTORY_CAPACITY, DEFAULT_GLOBAL_LEVEL, MAX_SINKS.

---

Usage Pattern

A subsystem that emits diagnostics:

```
import {
  createChannelLogger,
  LOG_CHANNEL,
} from './src/core/026_rnd_Logger.js';

const log = createChannelLogger(LOG_CHANNEL.SHADOWS);

log.info(() => `shadow atlas resized to ${newSize} pixels`);
log.warn('atlas page exhausted');
log.error(() => `failed to pack ${failedCount} of ${totalCount} pages`);
```

The thunk form matters on the hot path. In a tight loop where the message is expensive to build, pass a function so the string is only constructed when the level is enabled:

```
if (atlas.isDirty) {
  log.debug(() => `atlas dirty regions: ${atlas.countDirtyRegions()}`);
}
```

A debug HUD that pulls the last ten entries:

```
const entries = [];
logger.copyHistory(10, entries);
for (const entry of entries) {
  console.log(`[${LOG_CHANNEL_NAME[entry.channel]}] ${entry.message}`);
}
```

A crash reporter that installs a sink:

```
logger.addSink((entry) => {
  if (entry.level >= LOG_LEVEL.ERROR) {
    remoteEndpoint.send(entry);
  }
});
```

The App layer's thermal guard that wants warnings about throttling on its own channel:

```
logger.setChannelLevel(LOG_CHANNEL.QUALITY, LOG_LEVEL.DEBUG);
```

The logger is what makes the whole engine debuggable on a real device. On Android, the developer has no access to a debugger console during normal use, so the ring buffer is the primary diagnostic surface. When something goes wrong, the ring buffer contains the last N messages, and each message identifies its channel, its level, its frame number, and its wall-clock time. Combined with the profiler marker correlation, a frame drop can be traced to the exact log line that preceded it.

