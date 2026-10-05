API Documentation — src/core/005_rnd_FrameScheduler.js

File Purpose

This file is the multi-domain frame scheduler for the anime lighting stack. Where 003_rnd_Runtime.js holds a single FrameScheduler for the overall render-loop cadence, this module provides a separate, richer scheduler whose purpose is to drive each lighting subsystem at its own independent update rate.

Shadow atlas packing runs at 30 Hz. GI probe bakes run at 20 Hz. AO blur runs at 30 Hz. Environment palette solving runs at 8 Hz. The director hint emission runs at 60 Hz. Running all of those at the same frame rate would waste GPU and CPU cycles — the shadow atlas does not need to repack 60 times a second when the sun barely moved, and the environment palette does not need to update at 60 Hz when the day-cycle changes imperceptibly.

This module gives each lighting domain its own accumulator, its own budget, its own EMA, and its own adaptive downgrade/upgrade logic. The scheduler decides which domains fire on each frame based on integer accumulators, so there is no floating-point drift over long sessions.

It also provides thermal and battery bias hooks so an external controller (the App's thermal guard or battery guard) can downscale the entire lighting pipeline by lowering domain frequencies in one call.

---

Exported Constants

PERF_TIER

Type: string

Value: 'LOW' | 'MEDIUM' | 'HIGH' — cached from getPerfTier() at module load.

DOMAIN

Type: frozen enum

Maps symbolic names to integer indices.

Values:

· SIMULATION = 0
· LIGHTS = 1
· SHADOWS = 2
· GI = 3
· AO = 4
· ENVIRONMENT = 5
· INTERIOR = 6
· EXTERIOR = 7
· DIRECTOR = 8
· POST = 9
· COUNT = 10

DOMAIN_NAME

Type: frozen array

Maps integer indices back to strings: ['simulation', 'lights', 'shadows', 'gi', 'ao', 'environment', 'interior', 'exterior', 'director', 'post'].

DEFAULT_HZ

Type: frozen array of numbers

The default update frequency per domain, in Hertz:

· simulation — 60
· lights — 60
· shadows — 30
· gi — 20
· ao — 30
· environment — 8
· interior — 30
· exterior — 30
· director — 60
· post — 60

DEFAULT_BUDGET_MS

Type: frozen array of numbers

The default per-domain budget in milliseconds. Values depend on the device tier:

On HIGH: [4.0, 2.0, 6.0, 8.0, 3.0, 2.0, 2.0, 2.0, 1.0, 4.0].

On MEDIUM: [5.0, 2.5, 7.5, 10.0, 3.5, 2.5, 2.5, 2.5, 1.2, 5.0].

On LOW: [6.0, 3.0, 9.0, 12.0, 4.0, 3.0, 3.0, 3.0, 1.5, 6.0].

DEFAULT_ADAPTIVE

Type: frozen array of booleans

Which domains may adapt their frequency under load:

· simulation — true
· lights — true
· shadows — true
· gi — true
· ao — true
· environment — false
· interior — true
· exterior — true
· director — false
· post — true

MIN_HZ_FLOOR

Type: number

Value: 5

The lowest frequency any domain may be downscaled to.

MAX_HZ_CEIL

Type: number

Value: 120

The highest frequency any domain may be upscaled to.

STARVATION_FRAMES

Type: number

Value: 45

If a domain has been skipped for this many consecutive frames, the next tick() forces it to fire so a low-frequency domain is never fully starved on a frame-skipped loop.

EMA_ALPHA

Type: number

Value: 0.15

The smoothing factor for per-domain cost EMA.

DOWNGRADE_THRESHOLD

Type: number

Value: 1.05

A domain whose EMA exceeds its budget by 5 % is a candidate for downgrade.

UPGRADE_THRESHOLD

Type: number

Value: 0.70

A domain whose EMA is below 70 % of its budget is a candidate for upgrade.

DOWNGRADE_HOLD_FRAMES

Type: number

Value: 30

Consecutive frames over budget before the frequency is reduced.

UPGRADE_HOLD_FRAMES

Type: number

Value: 120

Consecutive frames under budget before the frequency is raised. This is much longer than the downgrade hold so the scheduler does not oscillate.

PRIORITY_SLACK

Type: number

Value: 0.20

Critical domains (LIGHTS, SHADOWS, POST) receive 20 % extra budget slack so a light-list rebuild can overrun slightly without triggering a downgrade.

CRITICAL_DOMAINS

Type: frozen array

[DOMAIN.LIGHTS, DOMAIN.SHADOWS, DOMAIN.POST].

---

Exported Class — DomainSlot

One instance per lighting domain. Owns the domain's frequency, budget, accumulators, EMA, and callback.

Constructor

```
new DomainSlot(index, name)
```

Parameters:

· index — the domain index (from DOMAIN).
· name — the domain name string.

Instance Properties

· index — integer domain index.
· name — string.
· enabled — 1 if enabled, 0 if not.
· baseHz — the default frequency for this domain, from DEFAULT_HZ.
· targetHz — the current frequency (may differ from baseHz under adaptation).
· minHz — the minimum allowed frequency, MIN_HZ_FLOOR.
· maxHz — the maximum allowed frequency, initialized to baseHz.
· budgetMs — the current budget in milliseconds (includes priority slack for critical domains).
· adaptive — 1 if adaptive, 0 otherwise.
· accumMs — accumulated milliseconds since the domain last fired.
· stepMs — the step size in milliseconds (1000 / targetHz).
· framesRun — total number of frames this domain has fired.
· framesSkip — total number of frames this domain has been skipped.
· framesOver — total number of fires that exceeded the budget.
· starve — consecutive frames skipped since the last fire.
· lastMs — cost in milliseconds of the last fire.
· lastEma — smoothed cost in milliseconds.
· peakMs — highest single cost recorded.
· callback — the registered callback function, or null.
· callbackCtx — the context object passed to the callback.
· priority — 1 for critical domains, 0 otherwise.
· _downgradeHold — internal countdown for downgrade hysteresis.
· _upgradeHold — internal countdown for upgrade hysteresis.

Instance Methods

setHz(hz)

Parameters: hz — the new frequency.

Returns: this.

Purpose: clamps hz to [minHz, maxHz], stores it in targetHz, and recomputes stepMs = 1000 / targetHz.

setBudgetMs(ms)

Parameters: ms — the new budget in milliseconds.

Returns: this.

Purpose: sets budgetMs after applying priority slack if this domain is critical.

setAdaptive(enabled)

Parameters: enabled — boolean.

Returns: this.

Purpose: toggles the adaptive flag. A non-adaptive domain ignores downgrade and upgrade decisions.

bind(fn, ctx)

Parameters:

· fn — the callback function (dt, elapsed, domainSlot) => void.
· ctx — the context object passed to fn.

Returns: this.

reset()

Returns: this.

Purpose: zeros every counter and both hold countdowns.

---

Exported Class — FrameScheduler

The main scheduler.

Constructor

```
new FrameScheduler(options = {})
```

Parameters:

· masterHz — the overall tick frequency. Default 60. Clamped to [15, 120].
· adaptive — whether per-domain adaptation is allowed. Default true.
· thermalBias — initial thermal bias, [0, 1]. Default 0.
· batteryBias — initial battery bias, [0, 1]. Default 0.
· hysteresis — whether to apply hold frames before downgrade/upgrade. Default true.
· cascadeLock — whether shadows, GI, and AO frequencies should be locked together during thermal/battery pressure. Default true.

Constructor work:

1. Allocates domains — an array of DOMAIN.COUNT DomainSlot instances.
2. Initializes master clock state: masterHz, masterStepMs, masterAccum.
3. Initializes overall counters: frame, elapsedMs, elapsed, lastDtMs, lastDtEmaMs.
4. Initializes biases and flags.
5. Allocates _frameMs (Float64Array of DOMAIN.COUNT) and _frameCalls (Uint32Array of DOMAIN.COUNT) for diagnostics.
6. Allocates _listeners (Map).

Instance Properties (Read-Only)

· domains — the array of DomainSlot instances.
· masterHz — the master tick frequency.
· frame — the frame counter.
· elapsedMs — total elapsed milliseconds.
· elapsed — total elapsed seconds.
· lastDtMs — the last delta in milliseconds.
· lastDtEmaMs — the smoothed delta in milliseconds.
· thermalBias — the current thermal bias.
· batteryBias — the current battery bias.
· adaptive — whether adaptation is enabled.
· hysteresis — whether hold frames are enforced.
· cascadeLock — whether shadows/GI/AO frequencies are locked.
· domainsFiredMask — a bitmask of which domains fired this frame.
· domainsSkipMask — a bitmask of which domains were skipped.
· domainsOverMask — a bitmask of which domains overran their budget.

Instance Methods

getDomain(index)

Parameters: index — the domain index.

Returns: the DomainSlot, or null if out of range.

setDomainHz(index, hz)

Parameters:

· index — the domain index.
· hz — the new frequency.

Returns: boolean — true on success.

setDomainBudget(index, ms)

Parameters:

· index — the domain index.
· ms — the new budget in milliseconds.

Returns: boolean.

setDomainEnabled(index, enabled)

Parameters:

· index — the domain index.
· enabled — boolean.

Returns: boolean.

bindDomain(index, fn, ctx)

Parameters:

· index — the domain index.
· fn — the callback function.
· ctx — the context object.

Returns: boolean.

setMasterHz(hz)

Parameters: hz — the master tick frequency.

Returns: this.

on(event, fn)

Parameters:

· event — one of 'tick', 'downgrade', 'upgrade'.
· fn — the callback.

Returns: unsubscribe function.

off(event, fn)

Returns: nothing.

setThermalBias(bias)

Parameters: bias — [0, 1], where 0 is nominal and 1 is critical.

Returns: this.

Purpose: stores the thermal bias, and if cascadeLock is on, applies the cascade lock to shadows, GI, and AO frequencies.

setBatteryBias(bias)

Parameters: bias — [0, 1].

Returns: this.

Purpose: same as setThermalBias but for battery pressure.

tick(dtMs, elapsedSec)

Parameters:

· dtMs — delta time in milliseconds.
· elapsedSec — total elapsed seconds.

Returns: the number of domains that fired this frame.

Purpose: the per-frame scheduler entry point.

Flow:

1. Increments frame, accumulates elapsedMs, updates lastDtEmaMs.
2. Accumulates masterAccum += dt. If it is below masterStepMs, the whole frame is skipped: sets domainsFiredMask = 0, domainsSkipMask = 0xFFFF, domainsOverMask = 0, and returns 0.
3. Otherwise subtracts masterStepMs from masterAccum.
4. Iterates every domain. For each enabled domain:
   · Adds dt to the domain's accumMs.
   · Increments framesSkip.
   · Checks the starvation guard: if starve >= STARVATION_FRAMES, forces the domain to fire.
   · If accumMs >= stepMs or the domain is being forced, fires the domain:
     · Subtracts stepMs (or zeroes accumMs if forced).
     · Times the callback with _now().
     · Calls callback(dt, elapsed, domainSlot) in try/catch.
     · Updates lastMs, lastEma, peakMs.
     · Resets framesSkip and starve.
     · Records the fire in _frameMs and _frameCalls.
     · Marks the domain over budget if the cost exceeds budgetMs.
     · Calls _adaptDomain(domain, dt).
     · Sets the domain's bit in firedMask.
   · Otherwise, increments starve and sets the domain's bit in skipMask.
5. Stores the three masks on the instance.
6. Emits tick with a summary.
7. Returns the number of domains that fired.

_adaptDomain(domain, dtMs)

Internal. Applies hysteresis logic:

· If domain.lastEma > domain.budgetMs * DOWNGRADE_THRESHOLD, increments _downgradeHold. When it reaches DOWNGRADE_HOLD_FRAMES, reduces targetHz by 15 % (floored to minHz), recomputes stepMs, emits downgrade, and resets _downgradeHold.
· Otherwise, resets _downgradeHold.
· If domain.lastEma < domain.budgetMs * UPGRADE_THRESHOLD and targetHz < maxHz, increments _upgradeHold. When it reaches UPGRADE_HOLD_FRAMES, increases targetHz by 10 % (capped at maxHz), recomputes stepMs, emits upgrade, and resets _upgradeHold.
· Otherwise, resets _upgradeHold.

_applyCascadeLock()

Internal. Reads the maximum of thermalBias and batteryBias and, based on the pressure, sets the target frequencies of SHADOWS, GI, and AO to scaled values:

· At zero pressure, restores 30 / 20 / 30 Hz.
· Otherwise, scales shadows down to 30 * (1 - pressure * 0.6), GI to 20 * (1 - pressure * 0.7), and AO to 30 * (1 - pressure * 0.6), each clamped to a floor of 5 / 3 / 5 Hz respectively.

_setTargetHz(index, hz)

Internal helper. Sets targetHz and stepMs for the domain at index.

reset()

Returns: this.

Purpose: resets every domain and every master clock counter.

suspend()

Returns: this.

Purpose: disables every domain. The tick call still runs but no domain will fire.

resume()

Returns: this.

Purpose: enables every domain.

dispose()

Returns: this.

Purpose: resets, clears listeners, and nulls all domain callbacks.

getStats()

Returns: an object with frame, elapsed, masterHz, lastDtMs, lastDtEmaMs, thermalBias, batteryBias, cascadeLock, the three masks, a domains array of per-domain stats, and perfTier.

getPressure()

Returns: a new Float32Array(DOMAIN.COUNT) where each entry is domain.lastEma / domain.budgetMs. Values in [0, 1] indicate headroom; values above 1 indicate overrun. Consumers use this to decide which domain to downgrade first.

---

Exported Functions

getDefaultScheduler()

Returns: the module-level singleton FrameScheduler, creating it on first call.

disposeDefaultScheduler()

Returns: nothing.

createFrameScheduler(options = {})

Returns: a new FrameScheduler.

bindLightingDomains(scheduler, callbacks = {})

Parameters:

· scheduler — a FrameScheduler instance.
· callbacks — an object with optional simulation, lights, shadows, gi, ao, environment, interior, exterior, director, post functions.

Returns: boolean — true on success.

Purpose: convenience helper that binds all ten domain callbacks in one call. Any callback omitted is bound to undefined, which the scheduler interprets as a no-op (the domain still fires and still charges its budget, but the callback is not invoked).

---

Default Export

The default export bundles: FrameScheduler, DomainSlot, createFrameScheduler, getDefaultScheduler, disposeDefaultScheduler, bindLightingDomains, DOMAIN, DOMAIN_NAME.

---

Usage Pattern

The EngineLoop calls frameScheduler.tick(dt, elapsed) once per frame. Downstream lighting systems register their per-frame work as domain callbacks:

```
import { getDefaultScheduler, DOMAIN, bindLightingDomains } from './src/core/005_rnd_FrameScheduler.js';

const scheduler = getDefaultScheduler();

bindLightingDomains(scheduler, {
  lights:      (dt, elapsed) => lightManager.update(dt, elapsed),
  shadows:     (dt, elapsed) => shadowSystem.update(dt, elapsed),
  gi:          (dt, elapsed) => giSystem.update(dt, elapsed),
  ao:          (dt, elapsed) => aoSystem.update(dt, elapsed),
  environment: (dt, elapsed) => environmentSystem.update(dt, elapsed),
});

// Thermal event from the App:
app.on('thermal', ({ state }) => {
  const bias = state === 'critical' ? 1.0
             : state === 'serious'  ? 0.8
             : state === 'fair'     ? 0.5
             : 0.0;
  scheduler.setThermalBias(bias);
});
```

The scheduler guarantees that lights fires before shadows at 60 / 30 Hz respectively, that gi fires at 20 Hz regardless of the master frame rate, and that when thermal pressure rises, the cascade lock reduces shadows, GI, and AO frequencies together — keeping the anime look coherent while freeing the CPU budget the device needs to avoid throttling.
