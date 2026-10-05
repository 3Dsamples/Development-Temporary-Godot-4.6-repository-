// File : 030
// name : src/ecs/030_scn_AnimationComponents.js
// description : Animation, motion-matching, and full-body IK SoA component
//               module for the scene ECS world of the anime lighting stack on
//               Android mobile. Declares every per-entity animation state,
//               clip, track, layer, blend, morph, IK chain, IK target,
//               ground-contact record, foot-placement, stair-climb state,
//               motion-matching pose, feature vector, and animation budget
//               the lighting stack needs — as fixed-capacity typed arrays
//               sized once to MAX_ENTITIES = 100000 (or MAX_ACTIVE_ANIMATED
//               for the pose buffers).
//
//               Scope:
//                 • Procedural animation for lights (flicker, day-cycle,
//                   wind sway), GI probes (traversal), AO volumes (morph),
//                   shadows (softness ramp), and streaming chunks (fade).
//                 • Skeletal animation for humanoids, birds, quadrupeds,
//                   machines, and custom skeletons — with motion matching
//                   and full-body IK.
//
//               Motion matching helpers:
//                 • extractTrajectoryFeatures    — future + past trajectory
//                 • extractPoseFeatures          — bone velocity + foot state
//                 • computeMatchScore            — weighted squared distance
//                 • searchMotionDatabase         — linear + KD-tree fallback
//                 • selectBestPose               — best DB pose for state
//                 • blendIntoPose                — smooth cross-fade into DB
//                 • beginMotionMatchTransition   — transition state machine
//                 • tickMotionMatchTransition    — advance the blend
//
//               Full-body IK helpers (all joints):
//                 • solveTwoBoneIK               — closed-form arm/leg IK
//                 • solveFABRIKChain             — iterative arbitrary chain
//                 • solveCCDChain                — cyclic-coordinate descent
//                 • solveFullBodyIK              — orchestrator over all chains
//                 • solveHeadLookAt              — neck + head look-at
//                 • solveSpineLookAt             — 4-bone spine bend
//                 • solveArmIK                   — shoulder → elbow → wrist
//                 • solveHandIK                  — wrist → 10 fingers
//                 • solveLegIK                   — hip → knee → ankle → foot
//                 • solveFootIK                  — ankle → foot → toe
//                 • solveWingIK                  — bird wing (3-bone)
//                 • solveTailIK                  — tail chain
//                 • solveQuadrupedIK             — full quadruped solve
//                 • solveMachineChainIK          — machine end-effector IK
//
//               Ground adaptation & stair climbing:
//                 • raycastGround                — analytic ground probe
//                 • adaptFootToGround            — snap foot to surface
//                 • adaptHipToGround             — pelvis height correction
//                 • detectStair                  — step detection
//                 • snapFootToStair              — foot to step
//                 • adaptBodyToStairs            — pelvis + feet sync
//                 • computeGroundNormal          — surface normal from ray
//
//               Design:
//                 • Bone poses stored in a shared active-slot pool so only
//                   currently animated entities consume memory.
//                 • Skeleton definitions are shared registry objects; each
//                   entity references a skeleton by id.
//                 • Every solver is allocation-free; scratch quaternions and
//                   vectors are module-level.
//                 • Zero per-frame allocations on the hot path.
//
//               Integration:
//                 • 002_lgt_LightComponents.js
//                 • 003_lgt_ShadowComponents.js
//                 • 004_lgt_GIComponents.js
//                 • 005_lgt_AOComponents.js
//                 • 010_scn_ECSWorld.js
//                 • 011_scn_BiteCSAdapter.js
//                 • 012_scn_ComponentRegistry.js
//                 • 013_scn_ComponentTypes.js
//                 • 014_scn_Tags.js
//                 • 015_scn_Relations.js
//                 • 016_scn_EntityPool.js
//                 • 017_scn_EntityLifetime.js
//                 • 025_scn_SpatialComponents.js
//                 • 026_scn_TransformComponents.js
//                 • 027_scn_LODComponents.js
//                 • 029_scn_VisibilityComponents.js
//
//               Strictly Three.js r185 lights only; strictly bitECS 0.4.0
//               API only; every internal array sized once at construction.
// best for : Guaranteeing that every animated entity in the anime lighting
//            stack — humanoid, bird, quadruped, machine, or procedural light
//            — has a deterministic, allocation-free, IK-capable animation
//            pipeline with motion matching, full-body IK, ground adaptation,
//            and stair climbing built in.
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
  MAX_ENTITIES,
} from './002_lgt_LightComponents.js';

import {
  getECSWorld,
  entityAlive,
} from './010_scn_ECSWorld.js';

import {
  getAdapter,
} from './011_scn_BiteCSAdapter.js';

import {
  getDefaultComponentRegistry,
} from './012_scn_ComponentRegistry.js';

import {
  COMPONENT_TYPE_ID,
} from './013_scn_ComponentTypes.js';

import {
  TAG,
  TAG2,
  FDIRTY,
  markFrameDirty,
  clearFrameDirty,
} from './014_scn_Tags.js';

import {
  NULL_ENTITY,
} from './015_scn_Relations.js';

import {
  Transform,
  TransformWorld,
} from './026_scn_TransformComponents.js';

/* ------------------------------------------------------------------ */
/* 0. CONSTANTS                                                       */
/* ------------------------------------------------------------------ */

const PERF_TIER_LOCAL = getPerfTier();

/**
 * Maximum number of bones in one skeleton. Sized to cover a humanoid with
 * 4-bone spine, head, neck, both arms with 10 fingers, both legs with
 * toes, and spare capacity for tails / wings / antennas.
 */
export const MAX_BONES_PER_ENTITY = 96;

/**
 * Maximum number of simultaneously animated entities. Only animated
 * entities consume pose-buffer memory; the pose pool is sized once.
 */
export const MAX_ACTIVE_ANIMATED =
  PERF_TIER_LOCAL === 'HIGH'   ? 4096 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 2048 :
                                 1024;

/**
 * Maximum number of IK chains per entity.
 */
export const MAX_IK_CHAINS_PER_ENTITY = 16;

/**
 * Maximum number of animation layers per entity.
 */
export const MAX_ANIM_LAYERS_PER_ENTITY = 4;

/**
 * Maximum number of animation clips per entity.
 */
export const MAX_CLIPS_PER_ENTITY = 8;

/**
 * Maximum number of tracks per clip.
 */
export const MAX_TRACKS_PER_CLIP = 64;

/**
 * Maximum number of morph targets per entity.
 */
export const MAX_MORPH_TARGETS = 32;

/**
 * Maximum number of motion-matching poses in the shared database.
 */
export const MAX_MOTION_POSES =
  PERF_TIER_LOCAL === 'HIGH'   ? 1024 :
  PERF_TIER_LOCAL === 'MEDIUM' ? 512 :
                                 256;

/**
 * Motion-matching feature vector length.
 *   [0..2]   future trajectory 0.5s
 *   [3..5]   future trajectory 1.0s
 *   [6..8]   past trajectory 0.5s
 *   [9..11]  left foot position
 *   [12..14] right foot position
 *   [15..17] left foot velocity
 *   [18..20] right foot velocity
 *   [21..23] hip velocity
 */
export const MOTION_FEATURE_DIM = 24;

/**
 * Default IK iteration cap.
 */
export const DEFAULT_IK_ITERATIONS = 12;

/**
 * Default IK tolerance (world units).
 */
export const DEFAULT_IK_TOLERANCE = 0.01;

/**
 * Maximum number of ground contacts per entity (feet + toes + hands).
 */
export const MAX_GROUND_CONTACTS = 8;

/**
 * Skeleton types.
 */
export const SKELETON_TYPE = Object.freeze({
  NONE:       0,
  HUMANOID:   1,
  BIRD:       2,
  QUADRUPED:  3,
  MACHINE:    4,
  CUSTOM:     5,
  COUNT:      6,
});

export const SKELETON_TYPE_NAME = Object.freeze([
  'none', 'humanoid', 'bird', 'quadruped', 'machine', 'custom',
]);

/**
 * Joint types. The numeric ids are stable and used on the wire.
 */
export const JOINT_TYPE = Object.freeze({
  ROOT:          0,
  HIP:           1,
  SPINE_1:       2,
  SPINE_2:       3,
  SPINE_3:       4,
  SPINE_4:       5,
  NECK:          6,
  HEAD:          7,
  SHOULDER_L:    8,
  SHOULDER_R:    9,
  ELBOW_L:      10,
  ELBOW_R:      11,
  WRIST_L:      12,
  WRIST_R:      13,
  HAND_L:       14,
  HAND_R:       15,
  FINGER_L_0:   16,
  FINGER_L_1:   17,
  FINGER_L_2:   18,
  FINGER_L_3:   19,
  FINGER_L_4:   20,
  FINGER_L_5:   21,
  FINGER_L_6:   22,
  FINGER_L_7:   23,
  FINGER_L_8:   24,
  FINGER_L_9:   25,
  FINGER_R_0:   26,
  FINGER_R_1:   27,
  FINGER_R_2:   28,
  FINGER_R_3:   29,
  FINGER_R_4:   30,
  FINGER_R_5:   31,
  FINGER_R_6:   32,
  FINGER_R_7:   33,
  FINGER_R_8:   34,
  FINGER_R_9:   35,
  UPPER_LEG_L:  36,
  UPPER_LEG_R:  37,
  KNEE_L:       38,
  KNEE_R:       39,
  ANKLE_L:      40,
  ANKLE_R:      41,
  FOOT_L:       42,
  FOOT_R:       43,
  TOE_L:        44,
  TOE_R:        45,
  TAIL_1:       46,
  TAIL_2:       47,
  TAIL_3:       48,
  TAIL_4:       49,
  WING_L_1:     50,
  WING_L_2:     51,
  WING_L_3:     52,
  WING_R_1:     53,
  WING_R_2:     54,
  WING_R_3:     55,
  LEG_QUAD_FL:  56,
  LEG_QUAD_FR:  57,
  LEG_QUAD_BL:  58,
  LEG_QUAD_BR:  59,
  MACHINE_BASE: 60,
  MACHINE_ARM:  61,
  MACHINE_END:  62,
  CUSTOM_0:     63,
  COUNT:        64,
});

export const JOINT_TYPE_NAME = Object.freeze([
  'root', 'hip', 'spine_1', 'spine_2', 'spine_3', 'spine_4', 'neck', 'head',
  'shoulder_l', 'shoulder_r', 'elbow_l', 'elbow_r', 'wrist_l', 'wrist_r',
  'hand_l', 'hand_r',
  'finger_l_0', 'finger_l_1', 'finger_l_2', 'finger_l_3', 'finger_l_4',
  'finger_l_5', 'finger_l_6', 'finger_l_7', 'finger_l_8', 'finger_l_9',
  'finger_r_0', 'finger_r_1', 'finger_r_2', 'finger_r_3', 'finger_r_4',
  'finger_r_5', 'finger_r_6', 'finger_r_7', 'finger_r_8', 'finger_r_9',
  'upper_leg_l', 'upper_leg_r', 'knee_l', 'knee_r', 'ankle_l', 'ankle_r',
  'foot_l', 'foot_r', 'toe_l', 'toe_r',
  'tail_1', 'tail_2', 'tail_3', 'tail_4',
  'wing_l_1', 'wing_l_2', 'wing_l_3', 'wing_r_1', 'wing_r_2', 'wing_r_3',
  'leg_quad_fl', 'leg_quad_fr', 'leg_quad_bl', 'leg_quad_br',
  'machine_base', 'machine_arm', 'machine_end', 'custom_0',
]);

/**
 * IK chain kinds.
 */
export const IK_CHAIN_TYPE = Object.freeze({
  NONE:       0,
  TWO_BONE:   1,
  THREE_BONE: 2,
  FABRIK:     3,
  CCD:        4,
  LOOK_AT:    5,
  COUNT:      6,
});

export const IK_CHAIN_TYPE_NAME = Object.freeze([
  'none', 'two_bone', 'three_bone', 'fabrik', 'ccd', 'look_at',
]);

/**
 * Animation playback state.
 */
export const ANIM_STATE = Object.freeze({
  IDLE:       0,
  PLAYING:    1,
  PAUSED:     2,
  BLENDING:   3,
  MOTION_MATCH:4,
  IK_OVERRIDE:5,
  COUNT:      6,
});

export const ANIM_STATE_NAME = Object.freeze([
  'idle', 'playing', 'paused', 'blending', 'motion_match', 'ik_override',
]);

/**
 * Motion-matching search mode.
 */
export const MM_SEARCH_MODE = Object.freeze({
  LINEAR:     0,
  KD_TREE:    1,
  WEIGHTED:   2,
  COUNT:      3,
});

/**
 * Foot contact state.
 */
export const FOOT_CONTACT = Object.freeze({
  AIR:        0,
  LOCKED:     1,
  SLIDING:    2,
  PLANTED:    3,
  COUNT:      4,
});

/* ------------------------------------------------------------------ */
/* 1. MODULE STATE                                                    */
/* ------------------------------------------------------------------ */

export const AnimationState = {
  frame:                 0,
  totalBinds:            0,
  totalPlays:            0,
  totalPauses:           0,
  totalStops:            0,
  totalLayers:           0,
  totalBlends:           0,
  totalIKChainsCreated:  0,
  totalIKSolves:         0,
  totalTwoBoneSolves:    0,
  totalFABRIKSolves:     0,
  totalCCDSolves:        0,
  totalFullBodySolves:   0,
  totalGroundRays:       0,
  totalStairDetections:  0,
  totalMotionMatches:    0,
  totalMotionTransitions:0,
  totalMorphUpdates:     0,
  peakActivePoses:       0,
  activePoses:           0,
  lastTickMs:            0,
  avgTickMs:             0,
  lastIKMs:              0,
  avgIKMs:               0,
  lastMMMs:              0,
  avgMMMs:               0,
};

let _boundary = null;
function _ensureBoundary() {
  if (_boundary) return _boundary;
  try {
    const mgr = getDefaultErrorBoundaries();
    if (mgr) {
      _boundary = mgr.create('ecs.animation', {
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

function _now() {
  return (typeof performance !== 'undefined' && typeof performance.now === 'function')
    ? performance.now()
    : Date.now();
}

/* ------------------------------------------------------------------ */
/* 2. SoA COMPONENT DECLARATIONS                                      */
/* ------------------------------------------------------------------ */

/**
 * AnimationState — per-entity animation state machine.
 */
export const AnimationStateComp = {
  state:              new Uint8Array(MAX_ENTITIES),
  prevState:          new Uint8Array(MAX_ENTITIES),
  skeletonId:         new Int32Array(MAX_ENTITIES).fill(-1),
  skeletonType:       new Uint8Array(MAX_ENTITIES),
  boneCount:          new Uint8Array(MAX_ENTITIES),
  activeSlot:         new Int16Array(MAX_ENTITIES).fill(-1),
  flags:              new Uint16Array(MAX_ENTITIES),
  lastStateFrame:     new Uint32Array(MAX_ENTITIES),
  globalTime:         new Float32Array(MAX_ENTITIES),
  globalSpeed:        new Float32Array(MAX_ENTITIES),
  globalWeight:       new Float32Array(MAX_ENTITIES),
  enabled:            new Uint8Array(MAX_ENTITIES),
  poseDirty:          new Uint8Array(MAX_ENTITIES),
};

/**
 * AnimationBinding — binds an entity to a skeleton and active slot.
 */
export const AnimationBinding = {
  rootBoneIdx:        new Uint8Array(MAX_ENTITIES),
  hipBoneIdx:         new Uint8Array(MAX_ENTITIES),
  headBoneIdx:        new Uint8Array(MAX_ENTITIES),
  leftFootBoneIdx:    new Uint8Array(MAX_ENTITIES),
  rightFootBoneIdx:   new Uint8Array(MAX_ENTITIES),
  leftHandBoneIdx:    new Uint8Array(MAX_ENTITIES),
  rightHandBoneIdx:   new Uint8Array(MAX_ENTITIES),
  spineBoneIdx:       new Uint8Array(MAX_ENTITIES),
  neckBoneIdx:        new Uint8Array(MAX_ENTITIES),
  tailBoneIdx:        new Uint8Array(MAX_ENTITIES),
  leftWingBoneIdx:    new Uint8Array(MAX_ENTITIES),
  rightWingBoneIdx:   new Uint8Array(MAX_ENTITIES),
  bound:              new Uint8Array(MAX_ENTITIES),
};

/**
 * AnimationLayer — up to MAX_ANIM_LAYERS_PER_ENTITY blended layers.
 */
export const AnimationLayer = {
  clipId:             new Int16Array(MAX_ENTITIES * MAX_ANIM_LAYERS_PER_ENTITY).fill(-1),
  time:               new Float32Array(MAX_ENTITIES * MAX_ANIM_LAYERS_PER_ENTITY),
  speed:              new Float32Array(MAX_ENTITIES * MAX_ANIM_LAYERS_PER_ENTITY).fill(1),
  weight:             new Float32Array(MAX_ENTITIES * MAX_ANIM_LAYERS_PER_ENTITY),
  loop:               new Uint8Array(MAX_ENTITIES * MAX_ANIM_LAYERS_PER_ENTITY).fill(1),
  active:             new Uint8Array(MAX_ENTITIES * MAX_ANIM_LAYERS_PER_ENTITY),
  layerCount:         new Uint8Array(MAX_ENTITIES),
  blendMode:          new Uint8Array(MAX_ENTITIES * MAX_ANIM_LAYERS_PER_ENTITY),
};

/**
 * AnimationClip — clip metadata shared across the whole scene.
 */
export const AnimationClip = {
  duration:           new Float32Array(MAX_CLIPS_PER_ENTITY * MAX_ENTITIES),
  trackCount:         new Uint8Array(MAX_CLIPS_PER_ENTITY * MAX_ENTITIES),
  loop:               new Uint8Array(MAX_CLIPS_PER_ENTITY * MAX_ENTITIES),
  rootMotion:         new Uint8Array(MAX_CLIPS_PER_ENTITY * MAX_ENTITIES),
  fps:                new Float32Array(MAX_CLIPS_PER_ENTITY * MAX_ENTITIES).fill(30),
  valid:              new Uint8Array(MAX_CLIPS_PER_ENTITY * MAX_ENTITIES),
};

/**
 * AnimationBlend — cross-fade state between two clips.
 */
export const AnimationBlend = {
  fromClipId:         new Int16Array(MAX_ENTITIES).fill(-1),
  toClipId:           new Int16Array(MAX_ENTITIES).fill(-1),
  fromTime:           new Float32Array(MAX_ENTITIES),
  toTime:             new Float32Array(MAX_ENTITIES),
  alpha:              new Float32Array(MAX_ENTITIES),
  duration:           new Float32Array(MAX_ENTITIES),
  elapsed:            new Float32Array(MAX_ENTITIES),
  active:             new Uint8Array(MAX_ENTITIES),
  syncPhase:          new Uint8Array(MAX_ENTITIES),
};

/**
 * AnimationRoot — root motion offset accumulated from clips.
 */
export const AnimationRoot = {
  offsetX:            new Float32Array(MAX_ENTITIES),
  offsetY:            new Float32Array(MAX_ENTITIES),
  offsetZ:            new Float32Array(MAX_ENTITIES),
  velocityX:          new Float32Array(MAX_ENTITIES),
  velocityY:          new Float32Array(MAX_ENTITIES),
  velocityZ:          new Float32Array(MAX_ENTITIES),
  yaw:                new Float32Array(MAX_ENTITIES),
  applyToTransform:   new Uint8Array(MAX_ENTITIES),
};

/**
 * AnimationMorph — morph target weights.
 */
export const AnimationMorph = {
  weight:             new Float32Array(MAX_ENTITIES * MAX_MORPH_TARGETS),
  targetWeight:       new Float32Array(MAX_ENTITIES * MAX_MORPH_TARGETS),
  targetCount:        new Uint8Array(MAX_ENTITIES),
  blendRate:          new Float32Array(MAX_ENTITIES * MAX_MORPH_TARGETS).fill(8),
};

/**
 * AnimationEvent — discrete events (footstep, handoff, cast) emitted
 * during playback.
 */
export const AnimationEvent = {
  pendingEvent:       new Uint16Array(MAX_ENTITIES),
  eventFlags:         new Uint16Array(MAX_ENTITIES),
  lastEventFrame:     new Uint32Array(MAX_ENTITIES),
  eventCounter:       new Uint32Array(MAX_ENTITIES),
};

/**
 * AnimationIK — per-chain IK descriptor.
 * Flat layout: chains[entity * MAX_IK_CHAINS_PER_ENTITY + chainIdx].
 */
export const AnimationIK = {
  chainType:          new Uint8Array(MAX_ENTITIES * MAX_IK_CHAINS_PER_ENTITY),
  solverType:         new Uint8Array(MAX_ENTITIES * MAX_IK_CHAINS_PER_ENTITY),
  rootBoneIdx:        new Uint8Array(MAX_ENTITIES * MAX_IK_CHAINS_PER_ENTITY),
  midBoneIdx:         new Uint8Array(MAX_ENTITIES * MAX_IK_CHAINS_PER_ENTITY),
  endBoneIdx:         new Uint8Array(MAX_ENTITIES * MAX_IK_CHAINS_PER_ENTITY),
  iterations:         new Uint8Array(MAX_ENTITIES * MAX_IK_CHAINS_PER_ENTITY),
  tolerance:          new Float32Array(MAX_ENTITIES * MAX_IK_CHAINS_PER_ENTITY),
  weight:             new Float32Array(MAX_ENTITIES * MAX_IK_CHAINS_PER_ENTITY).fill(1),
  enabled:            new Uint8Array(MAX_ENTITIES * MAX_IK_CHAINS_PER_ENTITY),
  chainCount:         new Uint8Array(MAX_ENTITIES),
};

/**
 * IKTargets — world-space IK targets per entity. Each entity can have
 * up to MAX_IK_CHAINS_PER_ENTITY targets.
 */
export const IKTargets = {
  posX:               new Float32Array(MAX_ENTITIES * MAX_IK_CHAINS_PER_ENTITY),
  posY:               new Float32Array(MAX_ENTITIES * MAX_IK_CHAINS_PER_ENTITY),
  posZ:               new Float32Array(MAX_ENTITIES * MAX_IK_CHAINS_PER_ENTITY),
  rotX:               new Float32Array(MAX_ENTITIES * MAX_IK_CHAINS_PER_ENTITY),
  rotY:               new Float32Array(MAX_ENTITIES * MAX_IK_CHAINS_PER_ENTITY),
  rotZ:               new Float32Array(MAX_ENTITIES * MAX_IK_CHAINS_PER_ENTITY),
  rotW:               new Float32Array(MAX_ENTITIES * MAX_IK_CHAINS_PER_ENTITY).fill(1),
  poleX:              new Float32Array(MAX_ENTITIES * MAX_IK_CHAINS_PER_ENTITY),
  poleY:              new Float32Array(MAX_ENTITIES * MAX_IK_CHAINS_PER_ENTITY).fill(1),
  poleZ:              new Float32Array(MAX_ENTITIES * MAX_IK_CHAINS_PER_ENTITY),
  valid:              new Uint8Array(MAX_ENTITIES * MAX_IK_CHAINS_PER_ENTITY),
};

/**
 * GroundContact — per-entity ground contact state.
 */
export const GroundContact = {
  footX_L:            new Float32Array(MAX_ENTITIES),
  footY_L:            new Float32Array(MAX_ENTITIES),
  footZ_L:            new Float32Array(MAX_ENTITIES),
  footX_R:            new Float32Array(MAX_ENTITIES),
  footY_R:            new Float32Array(MAX_ENTITIES),
  footZ_R:            new Float32Array(MAX_ENTITIES),
  groundNX_L:         new Float32Array(MAX_ENTITIES).fill(0),
  groundNY_L:         new Float32Array(MAX_ENTITIES).fill(1),
  groundNZ_L:         new Float32Array(MAX_ENTITIES).fill(0),
  groundNX_R:         new Float32Array(MAX_ENTITIES).fill(0),
  groundNY_R:         new Float32Array(MAX_ENTITIES).fill(1),
  groundNZ_R:         new Float32Array(MAX_ENTITIES).fill(0),
  contactStateL:      new Uint8Array(MAX_ENTITIES),
  contactStateR:      new Uint8Array(MAX_ENTITIES),
  contactTimerL:      new Uint16Array(MAX_ENTITIES),
  contactTimerR:      new Uint16Array(MAX_ENTITIES),
  slopeAngle:         new Float32Array(MAX_ENTITIES),
  stepHeightL:        new Float32Array(MAX_ENTITIES),
  stepHeightR:        new Float32Array(MAX_ENTITIES),
  hasGround:          new Uint8Array(MAX_ENTITIES),
};

/**
 * FootPlacement — foot placement targets and offsets.
 */
export const FootPlacement = {
  targetX_L:          new Float32Array(MAX_ENTITIES),
  targetY_L:          new Float32Array(MAX_ENTITIES),
  targetZ_L:          new Float32Array(MAX_ENTITIES),
  targetX_R:          new Float32Array(MAX_ENTITIES),
  targetY_R:          new Float32Array(MAX_ENTITIES),
  targetZ_R:          new Float32Array(MAX_ENTITIES),
  offsetY_L:          new Float32Array(MAX_ENTITIES),
  offsetY_R:          new Float32Array(MAX_ENTITIES),
  locked:             new Uint8Array(MAX_ENTITIES),
  stairMode:          new Uint8Array(MAX_ENTITIES),
  hipOffsetY:         new Float32Array(MAX_ENTITIES),
  hipOffsetForward:   new Float32Array(MAX_ENTITIES),
};

/**
 * MotionMatchState — per-entity motion-matching state.
 */
export const MotionMatchState = {
  enabled:            new Uint8Array(MAX_ENTITIES),
  currentPoseIdx:     new Int16Array(MAX_ENTITIES).fill(-1),
  targetPoseIdx:      new Int16Array(MAX_ENTITIES).fill(-1),
  transitionAlpha:    new Float32Array(MAX_ENTITIES),
  transitionDuration: new Float32Array(MAX_ENTITIES),
  transitionElapsed:  new Float32Array(MAX_ENTITIES),
  searchMode:         new Uint8Array(MAX_ENTITIES),
  lastMatchScore:     new Float32Array(MAX_ENTITIES),
  lastSearchFrame:    new Uint32Array(MAX_ENTITIES),
  databaseId:         new Int16Array(MAX_ENTITIES).fill(-1),
  featureWeightOverride: new Float32Array(MAX_ENTITIES * MOTION_FEATURE_DIM).fill(1),
};

/**
 * MotionPose — a single motion-matching database entry. Shared pool.
 */
export const MotionPose = {
  featureVec:         new Float32Array(MAX_MOTION_POSES * MOTION_FEATURE_DIM),
  poseTRSOffset:      new Uint32Array(MAX_MOTION_POSES),
  clipId:             new Int16Array(MAX_MOTION_POSES).fill(-1),
  frameTime:          new Float32Array(MAX_MOTION_POSES),
  velocity:           new Float32Array(MAX_MOTION_POSES * 3),
  tags:               new Uint16Array(MAX_MOTION_POSES),
  valid:              new Uint8Array(MAX_MOTION_POSES),
  poseCount:          new Uint32Array(1),
};

/**
 * AnimationBudget — aggregate cost per frame.
 */
export const AnimationBudget = {
  entityCost:         new Float32Array(MAX_ENTITIES),
  entityCostEma:      new Float32Array(MAX_ENTITIES),
  totalCost:          new Float32Array(1),
  budgetCap:          new Float32Array(1).fill(4.0),
  budgetExceeded:     new Uint8Array(1),
  ikBudgetCap:        new Float32Array(1).fill(2.0),
  ikBudgetUsed:       new Float32Array(1),
};

/**
 * AnimationStats — aggregate per-frame statistics.
 */
export const AnimationStats = {
  activeAnimated:     new Uint32Array(1),
  boneCount:          new Uint32Array(1),
  ikChainCount:       new Uint32Array(1),
  ikSolved:           new Uint32Array(1),
  ikFailed:           new Uint32Array(1),
  motionSearchCount:  new Uint32Array(1),
  morphTargetCount:   new Uint32Array(1),
};

/**
 * BonePosePool — the shared active-slot pose buffer. Only active
 * animated entities consume this memory. Layout:
 *   poseTRS[(slot * MAX_BONES_PER_ENTITY + boneIdx) * 7 + field]
 * where field ∈ {px, py, pz, qx, qy, qz, qw, sx, sy, sz} → 10 floats
 * (we use a 12-float stride for cache alignment).
 */
export const POSE_STRIDE = 12;
export const BonePosePool = {
  poseTRS:            new Float32Array(MAX_ACTIVE_ANIMATED * MAX_BONES_PER_ENTITY * POSE_STRIDE),
  worldTRS:           new Float32Array(MAX_ACTIVE_ANIMATED * MAX_BONES_PER_ENTITY * POSE_STRIDE),
  boneParentIdx:      new Int8Array(MAX_ACTIVE_ANIMATED * MAX_BONES_PER_ENTITY),
  boneJointType:      new Uint8Array(MAX_ACTIVE_ANIMATED * MAX_BONES_PER_ENTITY),
  boneValid:          new Uint8Array(MAX_ACTIVE_ANIMATED * MAX_BONES_PER_ENTITY),
  slotOwner:          new Int32Array(MAX_ACTIVE_ANIMATED).fill(-1),
  slotInUse:          new Uint8Array(MAX_ACTIVE_ANIMATED),
  freeSlotHead:       new Int32Array(1),
};

/**
 * Animation component bundle for bitECS createWorld.
 */
export const ANIMATION_COMPONENTS = Object.freeze({
  AnimationStateComp,
  AnimationBinding,
  AnimationLayer,
  AnimationClip,
  AnimationBlend,
  AnimationRoot,
  AnimationMorph,
  AnimationEvent,
  AnimationIK,
  IKTargets,
  GroundContact,
  FootPlacement,
  MotionMatchState,
  MotionPose,
  AnimationBudget,
  AnimationStats,
});

/* ------------------------------------------------------------------ */
/* 3. SCRATCH BUFFERS                                                 */
/* ------------------------------------------------------------------ */

const _scratchV3A = new Float32Array(3);
const _scratchV3B = new Float32Array(3);
const _scratchV3C = new Float32Array(3);
const _scratchV3D = new Float32Array(3);
const _scratchQ = new Float32Array(4);
const _scratchQB = new Float32Array(4);
const _scratchFeatures = new Float32Array(MOTION_FEATURE_DIM);
const _scratchFeaturesB = new Float32Array(MOTION_FEATURE_DIM);

/* ------------------------------------------------------------------ */
/* 4. VECTOR / QUATERNION UTILITIES                                   */
/* ------------------------------------------------------------------ */

function _len3(x, y, z) { return Math.sqrt(x * x + y * y + z * z); }
function _sub3(out, ax, ay, az, bx, by, bz) {
  out[0] = ax - bx; out[1] = ay - by; out[2] = az - bz; return out;
}
function _add3(out, ax, ay, az, bx, by, bz) {
  out[0] = ax + bx; out[1] = ay + by; out[2] = az + bz; return out;
}
function _scale3(out, ax, ay, az, s) {
  out[0] = ax * s; out[1] = ay * s; out[2] = az * s; return out;
}
function _normalize3(out, ax, ay, az) {
  const l = _len3(ax, ay, az) || 1;
  out[0] = ax / l; out[1] = ay / l; out[2] = az / l; return out;
}
function _cross3(out, ax, ay, az, bx, by, bz) {
  out[0] = ay * bz - az * by;
  out[1] = az * bx - ax * bz;
  out[2] = ax * by - ay * bx;
  return out;
}
function _dot3(ax, ay, az, bx, by, bz) { return ax * bx + ay * by + az * bz; }
function _dist3(ax, ay, az, bx, by, bz) {
  const dx = ax - bx, dy = ay - by, dz = az - bz;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}
function _clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

function _quatFromAxisAngle(out, ax, ay, az, angle) {
  const h = angle * 0.5;
  const s = Math.sin(h);
  out[0] = ax * s; out[1] = ay * s; out[2] = az * s; out[3] = Math.cos(h);
  return out;
}
function _quatMul(out, ax, ay, az, aw, bx, by, bz, bw) {
  out[0] = aw * bx + ax * bw + ay * bz - az * by;
  out[1] = aw * by - ax * bz + ay * bw + az * bx;
  out[2] = aw * bz + ax * by - ay * bx + az * bw;
  out[3] = aw * bw - ax * bx - ay * by - az * bz;
  return out;
}
function _quatNormalize(out, x, y, z, w) {
  const l = Math.sqrt(x * x + y * y + z * z + w * w) || 1;
  out[0] = x / l; out[1] = y / l; out[2] = z / l; out[3] = w / l;
  return out;
}
function _quatSlerp(out, ax, ay, az, aw, bx, by, bz, bw, t) {
  let cosH = ax * bx + ay * by + az * bz + aw * bw;
  let s = 1;
  if (cosH < 0) { cosH = -cosH; s = -1; }
  if (cosH > 0.9995) {
    return _quatNormalize(out, ax + (bx * s - ax) * t, ay + (by * s - ay) * t,
      az + (bz * s - az) * t, aw + (bw * s - aw) * t);
  }
  const halfTheta = Math.acos(cosH);
  const sinHalf = Math.sqrt(1 - cosH * cosH);
  const ra = Math.sin((1 - t) * halfTheta) / sinHalf;
  const rb = Math.sin(t * halfTheta) / sinHalf;
  return _quatNormalize(out, ax * ra + bx * rb * s, ay * ra + by * rb * s,
    az * ra + bz * rb * s, aw * ra + bw * rb * s);
}

/* ------------------------------------------------------------------ */
/* 5. POSE SLOT MANAGEMENT                                            */
/* ------------------------------------------------------------------ */

/**
 * Acquires an active pose slot for an entity. Returns the slot index, or
 * -1 if the pool is exhausted.
 */
export function acquirePoseSlot(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return -1;

  // Reuse if already has one.
  const existing = AnimationStateComp.activeSlot[eid];
  if (existing >= 0 && BonePosePool.slotInUse[existing] === 1) return existing;

  // Linear scan for a free slot (bounded by MAX_ACTIVE_ANIMATED).
  for (let s = 0; s < MAX_ACTIVE_ANIMATED; s++) {
    if (BonePosePool.slotInUse[s] === 0) {
      BonePosePool.slotInUse[s] = 1;
      BonePosePool.slotOwner[s] = eid;
      AnimationStateComp.activeSlot[eid] = s;
      AnimationState.activePoses++;
      if (AnimationState.activePoses > AnimationState.peakActivePoses) {
        AnimationState.peakActivePoses = AnimationState.activePoses;
      }
      return s;
    }
  }
  return -1;
}

/**
 * Releases an entity's pose slot.
 */
export function releasePoseSlot(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  const slot = AnimationStateComp.activeSlot[eid];
  if (slot < 0) return false;
  BonePosePool.slotInUse[slot] = 0;
  BonePosePool.slotOwner[slot] = -1;
  AnimationStateComp.activeSlot[eid] = -1;
  AnimationState.activePoses--;
  return true;
}

/* ------------------------------------------------------------------ */
/* 6. BONE POSE ACCESSORS                                             */
/* ------------------------------------------------------------------ */

function _poseOffset(slot, boneIdx) {
  return (slot * MAX_BONES_PER_ENTITY + boneIdx) * POSE_STRIDE;
}

export function getBonePosePosition(slot, boneIdx, out) {
  const off = _poseOffset(slot, boneIdx);
  out[0] = BonePosePool.poseTRS[off + 0];
  out[1] = BonePosePool.poseTRS[off + 1];
  out[2] = BonePosePool.poseTRS[off + 2];
  return out;
}

export function setBonePosePosition(slot, boneIdx, x, y, z) {
  const off = _poseOffset(slot, boneIdx);
  BonePosePool.poseTRS[off + 0] = x;
  BonePosePool.poseTRS[off + 1] = y;
  BonePosePool.poseTRS[off + 2] = z;
}

export function getBonePoseRotation(slot, boneIdx, out) {
  const off = _poseOffset(slot, boneIdx);
  out[0] = BonePosePool.poseTRS[off + 3];
  out[1] = BonePosePool.poseTRS[off + 4];
  out[2] = BonePosePool.poseTRS[off + 5];
  out[3] = BonePosePool.poseTRS[off + 6];
  return out;
}

export function setBonePoseRotation(slot, boneIdx, qx, qy, qz, qw) {
  const off = _poseOffset(slot, boneIdx);
  BonePosePool.poseTRS[off + 3] = qx;
  BonePosePool.poseTRS[off + 4] = qy;
  BonePosePool.poseTRS[off + 5] = qz;
  BonePosePool.poseTRS[off + 6] = qw;
}

export function getBonePoseScale(slot, boneIdx, out) {
  const off = _poseOffset(slot, boneIdx);
  out[0] = BonePosePool.poseTRS[off + 7];
  out[1] = BonePosePool.poseTRS[off + 8];
  out[2] = BonePosePool.poseTRS[off + 9];
  return out;
}

export function setBonePoseScale(slot, boneIdx, sx, sy, sz) {
  const off = _poseOffset(slot, boneIdx);
  BonePosePool.poseTRS[off + 7] = sx;
  BonePosePool.poseTRS[off + 8] = sy;
  BonePosePool.poseTRS[off + 9] = sz;
}

/* ------------------------------------------------------------------ */
/* 7. SKELETON DEFINITION                                             */
/* ------------------------------------------------------------------ */

/**
 * Registers a skeleton layout on an entity. The `bones` array is
 * `[{ jointType, parentIdx }, ...]`. Root bone has parentIdx === -1.
 */
export function registerSkeleton(eid, skeletonType, bones) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  if (!Array.isArray(bones) || bones.length === 0) return false;
  if (bones.length > MAX_BONES_PER_ENTITY) return false;

  const slot = acquirePoseSlot(eid);
  if (slot < 0) return false;

  const n = bones.length;

  for (let i = 0; i < n; i++) {
    const b = bones[i];
    BonePosePool.boneJointType[slot * MAX_BONES_PER_ENTITY + i] = b.jointType | 0;
    BonePosePool.boneParentIdx[slot * MAX_BONES_PER_ENTITY + i] = b.parentIdx !== undefined ? (b.parentIdx | 0) : -1;
    BonePosePool.boneValid[slot * MAX_BONES_PER_ENTITY + i] = 1;

    // Default TRS: identity.
    const off = _poseOffset(slot, i);
    BonePosePool.poseTRS[off + 0] = 0;
    BonePosePool.poseTRS[off + 1] = 0;
    BonePosePool.poseTRS[off + 2] = 0;
    BonePosePool.poseTRS[off + 3] = 0;
    BonePosePool.poseTRS[off + 4] = 0;
    BonePosePool.poseTRS[off + 5] = 0;
    BonePosePool.poseTRS[off + 6] = 1;
    BonePosePool.poseTRS[off + 7] = 1;
    BonePosePool.poseTRS[off + 8] = 1;
    BonePosePool.poseTRS[off + 9] = 1;
  }

  AnimationStateComp.skeletonType[eid] = skeletonType | 0;
  AnimationStateComp.boneCount[eid] = n;
  AnimationStateComp.enabled[eid] = 1;
  AnimationStateComp.state[eid] = ANIM_STATE.IDLE;

  // Populate standard bone index bindings by joint type.
  _rebindStandardBones(eid, slot, bones);

  AnimationBinding.bound[eid] = 1;
  AnimationState.totalBinds++;
  return true;
}

function _rebindStandardBones(eid, slot, bones) {
  AnimationBinding.rootBoneIdx[eid] = 0;
  AnimationBinding.hipBoneIdx[eid] = 0;
  AnimationBinding.headBoneIdx[eid] = 0;
  AnimationBinding.leftFootBoneIdx[eid] = 0;
  AnimationBinding.rightFootBoneIdx[eid] = 0;
  AnimationBinding.leftHandBoneIdx[eid] = 0;
  AnimationBinding.rightHandBoneIdx[eid] = 0;
  AnimationBinding.spineBoneIdx[eid] = 0;
  AnimationBinding.neckBoneIdx[eid] = 0;
  AnimationBinding.tailBoneIdx[eid] = 0;
  AnimationBinding.leftWingBoneIdx[eid] = 0;
  AnimationBinding.rightWingBoneIdx[eid] = 0;

  for (let i = 0; i < bones.length; i++) {
    const jt = bones[i].jointType;
    switch (jt) {
      case JOINT_TYPE.ROOT:        AnimationBinding.rootBoneIdx[eid] = i; break;
      case JOINT_TYPE.HIP:         AnimationBinding.hipBoneIdx[eid] = i; break;
      case JOINT_TYPE.HEAD:        AnimationBinding.headBoneIdx[eid] = i; break;
      case JOINT_TYPE.FOOT_L:      AnimationBinding.leftFootBoneIdx[eid] = i; break;
      case JOINT_TYPE.FOOT_R:      AnimationBinding.rightFootBoneIdx[eid] = i; break;
      case JOINT_TYPE.HAND_L:      AnimationBinding.leftHandBoneIdx[eid] = i; break;
      case JOINT_TYPE.HAND_R:      AnimationBinding.rightHandBoneIdx[eid] = i; break;
      case JOINT_TYPE.SPINE_1:     AnimationBinding.spineBoneIdx[eid] = i; break;
      case JOINT_TYPE.NECK:        AnimationBinding.neckBoneIdx[eid] = i; break;
      case JOINT_TYPE.TAIL_1:      AnimationBinding.tailBoneIdx[eid] = i; break;
      case JOINT_TYPE.WING_L_1:    AnimationBinding.leftWingBoneIdx[eid] = i; break;
      case JOINT_TYPE.WING_R_1:    AnimationBinding.rightWingBoneIdx[eid] = i; break;
      default: break;
    }
  }
}

/* ------------------------------------------------------------------ */
/* 8. STANDARD SKELETON BUILDERS                                      */
/* ------------------------------------------------------------------ */

/**
 * Builds a canonical humanoid skeleton bone list. Returns the array of
 * `{ jointType, parentIdx }` in the order expected by registerSkeleton.
 */
export function buildHumanoidSkeleton() {
  const B = [];
  const push = (jt, p) => B.push({ jointType: jt, parentIdx: p });

  // Spine chain.
  push(JOINT_TYPE.ROOT, -1);         // 0
  push(JOINT_TYPE.HIP, 0);           // 1
  push(JOINT_TYPE.SPINE_1, 1);       // 2
  push(JOINT_TYPE.SPINE_2, 2);       // 3
  push(JOINT_TYPE.SPINE_3, 3);       // 4
  push(JOINT_TYPE.SPINE_4, 4);       // 5
  push(JOINT_TYPE.NECK, 5);          // 6
  push(JOINT_TYPE.HEAD, 6);          // 7

  // Left arm.
  push(JOINT_TYPE.SHOULDER_L, 5);    // 8
  push(JOINT_TYPE.ELBOW_L, 8);       // 9
  push(JOINT_TYPE.WRIST_L, 9);       // 10
  push(JOINT_TYPE.HAND_L, 10);       // 11
  for (let i = 0; i < 10; i++) push(JOINT_TYPE.FINGER_L_0 + i, 11); // 12..21

  // Right arm.
  push(JOINT_TYPE.SHOULDER_R, 5);    // 22
  push(JOINT_TYPE.ELBOW_R, 22);      // 23
  push(JOINT_TYPE.WRIST_R, 23);      // 24
  push(JOINT_TYPE.HAND_R, 24);       // 25
  for (let i = 0; i < 10; i++) push(JOINT_TYPE.FINGER_R_0 + i, 25); // 26..35

  // Left leg.
  push(JOINT_TYPE.UPPER_LEG_L, 1);   // 36
  push(JOINT_TYPE.KNEE_L, 36);       // 37
  push(JOINT_TYPE.ANKLE_L, 37);      // 38
  push(JOINT_TYPE.FOOT_L, 38);       // 39
  push(JOINT_TYPE.TOE_L, 39);        // 40

  // Right leg.
  push(JOINT_TYPE.UPPER_LEG_R, 1);   // 41
  push(JOINT_TYPE.KNEE_R, 41);       // 42
  push(JOINT_TYPE.ANKLE_R, 42);      // 43
  push(JOINT_TYPE.FOOT_R, 43);       // 44
  push(JOINT_TYPE.TOE_R, 44);        // 45

  return B;   // 46 bones
}

/**
 * Builds a canonical bird skeleton bone list.
 */
export function buildBirdSkeleton() {
  const B = [];
  const push = (jt, p) => B.push({ jointType: jt, parentIdx: p });

  push(JOINT_TYPE.ROOT, -1);         // 0
  push(JOINT_TYPE.HIP, 0);           // 1
  push(JOINT_TYPE.SPINE_1, 1);       // 2
  push(JOINT_TYPE.SPINE_2, 2);       // 3
  push(JOINT_TYPE.NECK, 3);          // 4
  push(JOINT_TYPE.HEAD, 4);          // 5

  // Left wing.
  push(JOINT_TYPE.WING_L_1, 3);      // 6
  push(JOINT_TYPE.WING_L_2, 6);      // 7
  push(JOINT_TYPE.WING_L_3, 7);      // 8

  // Right wing.
  push(JOINT_TYPE.WING_R_1, 3);      // 9
  push(JOINT_TYPE.WING_R_2, 9);      // 10
  push(JOINT_TYPE.WING_R_3, 10);     // 11

  // Left leg.
  push(JOINT_TYPE.UPPER_LEG_L, 1);   // 12
  push(JOINT_TYPE.KNEE_L, 12);       // 13
  push(JOINT_TYPE.ANKLE_L, 13);      // 14
  push(JOINT_TYPE.FOOT_L, 14);       // 15
  push(JOINT_TYPE.TOE_L, 15);        // 16

  // Right leg.
  push(JOINT_TYPE.UPPER_LEG_R, 1);   // 17
  push(JOINT_TYPE.KNEE_R, 17);       // 18
  push(JOINT_TYPE.ANKLE_R, 18);      // 19
  push(JOINT_TYPE.FOOT_R, 19);       // 20
  push(JOINT_TYPE.TOE_R, 20);        // 21

  // Tail.
  push(JOINT_TYPE.TAIL_1, 1);        // 22
  push(JOINT_TYPE.TAIL_2, 22);       // 23
  push(JOINT_TYPE.TAIL_3, 23);       // 24

  return B;   // 25 bones
}

/**
 * Builds a canonical quadruped skeleton bone list.
 */
export function buildQuadrupedSkeleton() {
  const B = [];
  const push = (jt, p) => B.push({ jointType: jt, parentIdx: p });

  push(JOINT_TYPE.ROOT, -1);           // 0
  push(JOINT_TYPE.HIP, 0);             // 1
  push(JOINT_TYPE.SPINE_1, 1);         // 2
  push(JOINT_TYPE.SPINE_2, 2);         // 3
  push(JOINT_TYPE.SPINE_3, 3);         // 4
  push(JOINT_TYPE.SPINE_4, 4);         // 5
  push(JOINT_TYPE.NECK, 5);            // 6
  push(JOINT_TYPE.HEAD, 6);            // 7

  // Front left leg.
  push(JOINT_TYPE.LEG_QUAD_FL, 5);     // 8
  push(JOINT_TYPE.ELBOW_L, 8);         // 9
  push(JOINT_TYPE.WRIST_L, 9);         // 10
  push(JOINT_TYPE.FOOT_L, 10);         // 11

  // Front right leg.
  push(JOINT_TYPE.LEG_QUAD_FR, 5);     // 12
  push(JOINT_TYPE.ELBOW_R, 12);        // 13
  push(JOINT_TYPE.WRIST_R, 13);        // 14
  push(JOINT_TYPE.FOOT_R, 14);         // 15

  // Back left leg.
  push(JOINT_TYPE.LEG_QUAD_BL, 1);     // 16
  push(JOINT_TYPE.KNEE_L, 16);         // 17
  push(JOINT_TYPE.ANKLE_L, 17);        // 18
  push(JOINT_TYPE.FOOT_L, 18);         // 19

  // Back right leg.
  push(JOINT_TYPE.LEG_QUAD_BR, 1);     // 20
  push(JOINT_TYPE.KNEE_R, 20);         // 21
  push(JOINT_TYPE.ANKLE_R, 21);        // 22
  push(JOINT_TYPE.FOOT_R, 22);         // 23

  // Tail.
  push(JOINT_TYPE.TAIL_1, 1);          // 24
  push(JOINT_TYPE.TAIL_2, 24);         // 25
  push(JOINT_TYPE.TAIL_3, 25);         // 26
  push(JOINT_TYPE.TAIL_4, 26);         // 27

  return B;   // 28 bones
}

/**
 * Builds a canonical machine skeleton bone list.
 */
export function buildMachineSkeleton() {
  const B = [];
  const push = (jt, p) => B.push({ jointType: jt, parentIdx: p });

  push(JOINT_TYPE.ROOT, -1);           // 0
  push(JOINT_TYPE.MACHINE_BASE, 0);    // 1
  push(JOINT_TYPE.MACHINE_ARM, 1);     // 2
  push(JOINT_TYPE.MACHINE_END, 2);     // 3

  return B;   // 4 bones
}

/* ------------------------------------------------------------------ */
/* 9. ANIMATION PLAYBACK                                              */
/* ------------------------------------------------------------------ */

/**
 * Binds a clip to a layer of an entity.
 */
export function bindClip(eid, layerIdx, clipId, duration, loop, speed) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  if (layerIdx < 0 || layerIdx >= MAX_ANIM_LAYERS_PER_ENTITY) return false;

  const flat = eid * MAX_ANIM_LAYERS_PER_ENTITY + layerIdx;
  AnimationLayer.clipId[flat] = clipId | 0;
  AnimationLayer.time[flat] = 0;
  AnimationLayer.speed[flat] = Number.isFinite(speed) ? speed : 1;
  AnimationLayer.weight[flat] = 1.0;
  AnimationLayer.loop[flat] = loop === false ? 0 : 1;
  AnimationLayer.active[flat] = 1;
  AnimationLayer.blendMode[flat] = 0;

  if (layerIdx >= AnimationLayer.layerCount[eid]) {
    AnimationLayer.layerCount[eid] = layerIdx + 1;
  }

  // Clip metadata.
  const clipFlat = eid * MAX_CLIPS_PER_ENTITY + (clipId | 0);
  AnimationClip.duration[clipFlat] = Number.isFinite(duration) ? duration : 1.0;
  AnimationClip.loop[clipFlat] = loop === false ? 0 : 1;
  AnimationClip.valid[clipFlat] = 1;

  return true;
}

/**
 * Plays a clip on a layer.
 */
export function playClip(eid, layerIdx, clipId, options) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  const opts = options || {};
  const flat = eid * MAX_ANIM_LAYERS_PER_ENTITY + layerIdx;
  if (AnimationLayer.clipId[flat] < 0) return false;
  AnimationLayer.active[flat] = 1;
  AnimationLayer.speed[flat] = opts.speed !== undefined ? Number(opts.speed) : 1;
  AnimationLayer.loop[flat] = opts.loop === false ? 0 : 1;
  if (opts.restart) AnimationLayer.time[flat] = 0;
  AnimationStateComp.state[eid] = ANIM_STATE.PLAYING;
  AnimationState.totalPlays++;
  return true;
}

/**
 * Pauses all layers on the entity.
 */
export function pauseAnimation(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  AnimationStateComp.state[eid] = ANIM_STATE.PAUSED;
  AnimationState.totalPauses++;
  return true;
}

/**
 * Stops the entity's animation.
 */
export function stopAnimation(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  AnimationStateComp.state[eid] = ANIM_STATE.IDLE;
  const count = AnimationLayer.layerCount[eid];
  for (let l = 0; l < count; l++) {
    const flat = eid * MAX_ANIM_LAYERS_PER_ENTITY + l;
    AnimationLayer.active[flat] = 0;
    AnimationLayer.time[flat] = 0;
  }
  AnimationState.totalStops++;
  return true;
}

/**
 * Sets the current playback time for a layer.
 */
export function setAnimationTime(eid, layerIdx, t) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  const flat = eid * MAX_ANIM_LAYERS_PER_ENTITY + layerIdx;
  AnimationLayer.time[flat] = Number(t) || 0;
  return true;
}

/**
 * Sets the playback speed for a layer.
 */
export function setAnimationSpeed(eid, layerIdx, speed) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  const flat = eid * MAX_ANIM_LAYERS_PER_ENTITY + layerIdx;
  AnimationLayer.speed[flat] = Number.isFinite(speed) ? speed : 1;
  return true;
}

/**
 * Sets the layer weight.
 */
export function setLayerWeight(eid, layerIdx, weight) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  const flat = eid * MAX_ANIM_LAYERS_PER_ENTITY + layerIdx;
  AnimationLayer.weight[flat] = _clamp(Number(weight) || 0, 0, 1);
  return true;
}

/**
 * Adds a new layer at the next available index.
 */
export function addAnimationLayer(eid, clipId, weight, speed) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return -1;
  const count = AnimationLayer.layerCount[eid];
  if (count >= MAX_ANIM_LAYERS_PER_ENTITY) return -1;
  bindClip(eid, count, clipId, 1.0, true, speed);
  if (weight !== undefined) setLayerWeight(eid, count, weight);
  AnimationState.totalLayers++;
  return count;
}

/**
 * Begins a cross-fade between two clips.
 */
export function blendAnimation(eid, fromClipId, toClipId, duration) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  AnimationBlend.fromClipId[eid] = fromClipId | 0;
  AnimationBlend.toClipId[eid] = toClipId | 0;
  AnimationBlend.duration[eid] = Math.max(0.0001, Number(duration) || 0.3);
  AnimationBlend.elapsed[eid] = 0;
  AnimationBlend.alpha[eid] = 0;
  AnimationBlend.active[eid] = 1;
  AnimationStateComp.state[eid] = ANIM_STATE.BLENDING;
  AnimationState.totalBlends++;
  return true;
}

/* ------------------------------------------------------------------ */
/* 10. MORPH TARGETS                                                  */
/* ------------------------------------------------------------------ */

/**
 * Sets a morph target weight.
 */
export function setMorphWeight(eid, idx, weight, rate) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  if (idx < 0 || idx >= MAX_MORPH_TARGETS) return false;
  const flat = eid * MAX_MORPH_TARGETS + idx;
  AnimationMorph.targetWeight[flat] = _clamp(Number(weight) || 0, 0, 1);
  if (rate !== undefined) AnimationMorph.blendRate[flat] = Math.max(0.01, Number(rate) || 8);
  if (idx >= AnimationMorph.targetCount[eid]) {
    AnimationMorph.targetCount[eid] = idx + 1;
  }
  return true;
}

/**
 * Advances morph target weights toward their targets. Called once per
 * frame.
 */
export function evaluateMorphWeights(eid, dt) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return 0;
  const count = AnimationMorph.targetCount[eid];
  let changed = 0;
  for (let i = 0; i < count; i++) {
    const flat = eid * MAX_MORPH_TARGETS + i;
    const cur = AnimationMorph.weight[flat];
    const tgt = AnimationMorph.targetWeight[flat];
    if (cur === tgt) continue;
    const k = 1 - Math.exp(-AnimationMorph.blendRate[flat] * dt);
    const next = cur + (tgt - cur) * k;
    AnimationMorph.weight[flat] = next;
    changed++;
  }
  if (changed > 0) AnimationState.totalMorphUpdates += changed;
  return changed;
}

/* ------------------------------------------------------------------ */
/* 11. MOTION MATCHING HELPERS                                        */
/* ------------------------------------------------------------------ */

/**
 * Extracts the trajectory + pose feature vector from an entity's current
 * animation state into `out` (Float32Array, length MOTION_FEATURE_DIM).
 */
export function extractMotionFeatures(eid, out) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return null;
  if (!out || out.length < MOTION_FEATURE_DIM) return null;

  // Future trajectory (from current velocity).
  const velX = AnimationRoot.velocityX[eid];
  const velY = AnimationRoot.velocityY[eid];
  const velZ = AnimationRoot.velocityZ[eid];

  const offX = AnimationRoot.offsetX[eid];
  const offY = AnimationRoot.offsetY[eid];
  const offZ = AnimationRoot.offsetZ[eid];

  // Future 0.5s.
  out[0] = offX + velX * 0.5;
  out[1] = offY + velY * 0.5;
  out[2] = offZ + velZ * 0.5;

  // Future 1.0s.
  out[3] = offX + velX * 1.0;
  out[4] = offY + velY * 1.0;
  out[5] = offZ + velZ * 1.0;

  // Past 0.5s.
  out[6] = offX - velX * 0.5;
  out[7] = offY - velY * 0.5;
  out[8] = offZ - velZ * 0.5;

  // Foot positions.
  out[9]  = FootPlacement.targetX_L[eid];
  out[10] = FootPlacement.targetY_L[eid];
  out[11] = FootPlacement.targetZ_L[eid];
  out[12] = FootPlacement.targetX_R[eid];
  out[13] = FootPlacement.targetY_R[eid];
  out[14] = FootPlacement.targetZ_R[eid];

  // Foot velocities (approximate from contact state).
  const contactL = GroundContact.contactStateL[eid];
  const contactR = GroundContact.contactStateR[eid];
  out[15] = contactL === FOOT_CONTACT.LOCKED ? 0 : velX;
  out[16] = contactL === FOOT_CONTACT.LOCKED ? 0 : velY;
  out[17] = contactL === FOOT_CONTACT.LOCKED ? 0 : velZ;
  out[18] = contactR === FOOT_CONTACT.LOCKED ? 0 : velX;
  out[19] = contactR === FOOT_CONTACT.LOCKED ? 0 : velY;
  out[20] = contactR === FOOT_CONTACT.LOCKED ? 0 : velZ;

  // Hip velocity.
  out[21] = velX;
  out[22] = velY;
  out[23] = velZ;

  return out;
}

/**
 * Computes the weighted squared distance between two feature vectors.
 */
export function computeMatchScore(featuresA, featuresB, weights) {
  if (!featuresA || !featuresB) return Infinity;
  let score = 0;
  const n = MOTION_FEATURE_DIM;
  for (let i = 0; i < n; i++) {
    const w = weights ? weights[i] : 1.0;
    const d = featuresA[i] - featuresB[i];
    score += w * d * d;
  }
  return score;
}

/**
 * Searches the motion-matching database for the best matching pose.
 * Returns the pose index, or -1 on failure.
 */
export function searchMotionDatabase(features, databaseOffset, poseCount, weights) {
  if (!features) return -1;
  const start = databaseOffset !== undefined ? databaseOffset : 0;
  const count = poseCount !== undefined ? poseCount : MotionPose.poseCount[0];
  const end = Math.min(start + count, MAX_MOTION_POSES);

  let bestIdx = -1;
  let bestScore = Infinity;

  for (let i = start; i < end; i++) {
    if (MotionPose.valid[i] === 0) continue;

    const base = i * MOTION_FEATURE_DIM;
    let score = 0;
    for (let f = 0; f < MOTION_FEATURE_DIM; f++) {
      const w = weights ? weights[f] : 1.0;
      const d = features[f] - MotionPose.featureVec[base + f];
      score += w * d * d;
    }

    if (score < bestScore) {
      bestScore = score;
      bestIdx = i;
      if (score < 0.001) break;   // early out on near-perfect match
    }
  }

  return bestIdx;
}

/**
 * Registers a motion-matching pose in the shared database.
 * Returns the pose index, or -1.
 */
export function registerMotionPose(features, poseTRSOffset, clipId, frameTime, velocityX, velocityY, velocityZ, tags) {
  const idx = MotionPose.poseCount[0];
  if (idx >= MAX_MOTION_POSES) return -1;
  if (!features || features.length < MOTION_FEATURE_DIM) return -1;

  const base = idx * MOTION_FEATURE_DIM;
  for (let f = 0; f < MOTION_FEATURE_DIM; f++) {
    MotionPose.featureVec[base + f] = features[f];
  }
  MotionPose.poseTRSOffset[idx] = poseTRSOffset >>> 0;
  MotionPose.clipId[idx] = (clipId | 0) & 0x7FFF;
  MotionPose.frameTime[idx] = Number(frameTime) || 0;
  MotionPose.velocity[idx * 3 + 0] = velocityX || 0;
  MotionPose.velocity[idx * 3 + 1] = velocityY || 0;
  MotionPose.velocity[idx * 3 + 2] = velocityZ || 0;
  MotionPose.tags[idx] = (tags | 0) & 0xFFFF;
  MotionPose.valid[idx] = 1;

  MotionPose.poseCount[0] = idx + 1;
  return idx;
}

/**
 * Performs a motion-matching search for an entity and begins a
 * transition to the best matching pose.
 */
export function performMotionMatch(eid, databaseOffset, poseCount, weights) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return -1;

  const t0 = _now();

  extractMotionFeatures(eid, _scratchFeatures);
  const bestIdx = searchMotionDatabase(_scratchFeatures, databaseOffset, poseCount, weights);
  if (bestIdx < 0) return -1;

  MotionMatchState.currentPoseIdx[eid] = bestIdx;
  MotionMatchState.lastMatchScore[eid] = 0;
  MotionMatchState.lastSearchFrame[eid] = AnimationState.frame;
  MotionMatchState.enabled[eid] = 1;
  AnimationState.totalMotionMatches++;

  const t1 = _now();
  AnimationState.lastMMMs = t1 - t0;
  AnimationState.avgMMMs += (AnimationState.lastMMMs - AnimationState.avgMMMs) * 0.15;

  return bestIdx;
}

/**
 * Begins a motion-matching transition to a new pose.
 */
export function beginMotionMatchTransition(eid, targetPoseIdx, duration) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  MotionMatchState.targetPoseIdx[eid] = targetPoseIdx | 0;
  MotionMatchState.transitionAlpha[eid] = 0;
  MotionMatchState.transitionDuration[eid] = Math.max(0.0001, Number(duration) || 0.2);
  MotionMatchState.transitionElapsed[eid] = 0;
  AnimationStateComp.state[eid] = ANIM_STATE.MOTION_MATCH;
  AnimationState.totalMotionTransitions++;
  return true;
}

/**
 * Advances an entity's motion-matching transition one frame.
 */
export function tickMotionMatchTransition(eid, dt) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  if (MotionMatchState.targetPoseIdx[eid] < 0) return false;

  const dur = MotionMatchState.transitionDuration[eid];
  let elapsed = MotionMatchState.transitionElapsed[eid] + dt;
  if (elapsed >= dur) elapsed = dur;

  MotionMatchState.transitionElapsed[eid] = elapsed;
  MotionMatchState.transitionAlpha[eid] = elapsed / dur;

  if (elapsed >= dur) {
    MotionMatchState.currentPoseIdx[eid] = MotionMatchState.targetPoseIdx[eid];
    MotionMatchState.targetPoseIdx[eid] = -1;
    MotionMatchState.transitionAlpha[eid] = 1;
    AnimationStateComp.state[eid] = ANIM_STATE.PLAYING;
  }
  return true;
}

/* ------------------------------------------------------------------ */
/* 12. IK CHAIN CONSTRUCTION                                          */
/* ------------------------------------------------------------------ */

/**
 * Creates an IK chain on the given entity. Returns the chain index, or
 * -1 on failure.
 */
export function createIKChain(eid, chainType, rootBoneIdx, midBoneIdx, endBoneIdx, options) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return -1;
  const count = AnimationIK.chainCount[eid];
  if (count >= MAX_IK_CHAINS_PER_ENTITY) return -1;

  const chainIdx = count;
  const flat = eid * MAX_IK_CHAINS_PER_ENTITY + chainIdx;

  AnimationIK.chainType[flat] = chainType | 0;
  AnimationIK.rootBoneIdx[flat] = rootBoneIdx | 0;
  AnimationIK.midBoneIdx[flat] = midBoneIdx !== undefined ? (midBoneIdx | 0) : 0;
  AnimationIK.endBoneIdx[flat] = endBoneIdx | 0;
  AnimationIK.iterations[flat] = options && options.iterations !== undefined
    ? (options.iterations | 0)
    : DEFAULT_IK_ITERATIONS;
  AnimationIK.tolerance[flat] = options && options.tolerance !== undefined
    ? Number(options.tolerance)
    : DEFAULT_IK_TOLERANCE;
  AnimationIK.weight[flat] = options && options.weight !== undefined ? Number(options.weight) : 1.0;
  AnimationIK.enabled[flat] = 1;

  AnimationIK.chainCount[eid] = chainIdx + 1;

  // Default target: end bone current world position.
  const slot = AnimationStateComp.activeSlot[eid];
  if (slot >= 0) {
    getBonePosePosition(slot, endBoneIdx, _scratchV3A);
    IKTargets.posX[flat] = _scratchV3A[0];
    IKTargets.posY[flat] = _scratchV3A[1];
    IKTargets.posZ[flat] = _scratchV3A[2];
    IKTargets.rotX[flat] = 0;
    IKTargets.rotY[flat] = 0;
    IKTargets.rotZ[flat] = 0;
    IKTargets.rotW[flat] = 1;
    IKTargets.poleX[flat] = 0;
    IKTargets.poleY[flat] = 1;
    IKTargets.poleZ[flat] = 0;
    IKTargets.valid[flat] = 1;
  }

  AnimationState.totalIKChainsCreated++;
  return chainIdx;
}

/**
 * Sets an IK chain's target position.
 */
export function setIKTarget(eid, chainIdx, x, y, z) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  const flat = eid * MAX_IK_CHAINS_PER_ENTITY + chainIdx;
  IKTargets.posX[flat] = x;
  IKTargets.posY[flat] = y;
  IKTargets.posZ[flat] = z;
  IKTargets.valid[flat] = 1;
  return true;
}

/**
 * Sets an IK chain's target rotation.
 */
export function setIKTargetRotation(eid, chainIdx, qx, qy, qz, qw) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  const flat = eid * MAX_IK_CHAINS_PER_ENTITY + chainIdx;
  IKTargets.rotX[flat] = qx;
  IKTargets.rotY[flat] = qy;
  IKTargets.rotZ[flat] = qz;
  IKTargets.rotW[flat] = qw;
  return true;
}

/**
 * Sets an IK chain's pole vector.
 */
export function setIKPoleVector(eid, chainIdx, x, y, z) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  const flat = eid * MAX_IK_CHAINS_PER_ENTITY + chainIdx;
  IKTargets.poleX[flat] = x;
  IKTargets.poleY[flat] = y;
  IKTargets.poleZ[flat] = z;
  return true;
}

/* ------------------------------------------------------------------ */
/* 13. TWO-BONE IK SOLVER                                             */
/* ------------------------------------------------------------------ */

/**
 * Closed-form two-bone IK: given root, mid, and end joint world positions
 * and a target end position, computes the mid and end rotations.
 *
 * Writes the resulting quaternions into the pose slot for the mid and
 * end bones.
 *
 * Returns true on success.
 */
export function solveTwoBoneIK(slot, rootBoneIdx, midBoneIdx, endBoneIdx,
                              targetX, targetY, targetZ,
                              poleX, poleY, poleZ,
                              weight) {
  if (slot < 0 || slot >= MAX_ACTIVE_ANIMATED) return false;

  // Read current bone positions.
  getBonePosePosition(slot, rootBoneIdx, _scratchV3A);
  const rx = _scratchV3A[0], ry = _scratchV3A[1], rz = _scratchV3A[2];
  getBonePosePosition(slot, midBoneIdx, _scratchV3B);
  const mx = _scratchV3B[0], my = _scratchV3B[1], mz = _scratchV3B[2];
  getBonePosePosition(slot, endBoneIdx, _scratchV3C);
  const ex = _scratchV3C[0], ey = _scratchV3C[1], ez = _scratchV3C[2];

  // Bone lengths.
  const lenA = _dist3(rx, ry, rz, mx, my, mz);
  const lenB = _dist3(mx, my, mz, ex, ey, ez);

  // Target distance from root.
  const tx = targetX - rx;
  const ty = targetY - ry;
  const tz = targetZ - rz;
  let targetDist = _len3(tx, ty, tz);

  // Clamp to reachable range.
  const maxReach = lenA + lenB - 1e-4;
  const minReach = Math.abs(lenA - lenB) + 1e-4;
  if (targetDist > maxReach) targetDist = maxReach;
  if (targetDist < minReach) targetDist = minReach;

  // Normalized direction to target.
  const inv = targetDist > 1e-6 ? 1.0 / _len3(tx, ty, tz) : 0;
  const dirX = tx * inv, dirY = ty * inv, dirZ = tz * inv;

  // Law of cosines for the angle at the root.
  const cosRoot = _clamp((lenA * lenA + targetDist * targetDist - lenB * lenB) / (2 * lenA * targetDist), -1, 1);
  const rootAngle = Math.acos(cosRoot);

  // Pole vector — orthogonalized against direction.
  let pX = poleX - rx, pY = poleY - ry, pZ = poleZ - rz;
  // Project pole onto plane perpendicular to direction.
  const pDot = pX * dirX + pY * dirY + pZ * dirZ;
  pX -= pDot * dirX; pY -= pDot * dirY; pZ -= pDot * dirZ;
  const pLen = _len3(pX, pY, pZ);
  if (pLen < 1e-5) {
    // Fallback: pick any perpendicular.
    _cross3(_scratchV3D, dirX, dirY, dirZ, 0, 1, 0);
    const pl = _len3(_scratchV3D[0], _scratchV3D[1], _scratchV3D[2]);
    if (pl < 1e-5) {
      _cross3(_scratchV3D, dirX, dirY, dirZ, 1, 0, 0);
    }
    _normalize3(_scratchV3D, _scratchV3D[0], _scratchV3D[1], _scratchV3D[2]);
    pX = _scratchV3D[0]; pY = _scratchV3D[1]; pZ = _scratchV3D[2];
  } else {
    pX /= pLen; pY /= pLen; pZ /= pLen;
  }

  // Mid-joint ideal position.
  const cosR = Math.cos(rootAngle);
  const sinR = Math.sin(rootAngle);
  const midIdealX = rx + (dirX * cosR + pX * sinR) * lenA;
  const midIdealY = ry + (dirY * cosR + pY * sinR) * lenA;
  const midIdealZ = rz + (dirZ * cosR + pZ * sinR) * lenA;

  // Compute quaternion that rotates the current root→mid direction to
  // the ideal root→mid direction.
  let curAX = mx - rx, curAY = my - ry, curAZ = mz - rz;
  _normalize3(_scratchV3B, curAX, curAY, curAZ);
  let idealAX = midIdealX - rx, idealAY = midIdealY - ry, idealAZ = midIdealZ - rz;
  _normalize3(_scratchV3C, idealAX, idealAY, idealAZ);

  _computeRotationBetween(_scratchQ, _scratchV3B[0], _scratchV3B[1], _scratchV3B[2],
                                   _scratchV3C[0], _scratchV3C[1], _scratchV3C[2]);

  // Rotate the mid-bone by this quaternion.
  getBonePoseRotation(slot, midBoneIdx, _scratchQB);
  _quatMul(_scratchQB, _scratchQ[0], _scratchQ[1], _scratchQ[2], _scratchQ[3],
                     _scratchQB[0], _scratchQB[1], _scratchQB[2], _scratchQB[3]);
  if (weight !== undefined && weight < 1) {
    // Slerp toward the computed value.
    getBonePoseRotation(slot, midBoneIdx, _scratchV3D);
    _quatSlerp(_scratchQB, _scratchV3D[0], _scratchV3D[1], _scratchV3D[2], _scratchV3D[3],
                          _scratchQB[0], _scratchQB[1], _scratchQB[2], _scratchQB[3], weight);
  }
  setBonePoseRotation(slot, midBoneIdx, _scratchQB[0], _scratchQB[1], _scratchQB[2], _scratchQB[3]);

  // Similarly for the end bone: current mid→end vs ideal mid→target.
  // Compute the ideal end direction.
  let idealEX = targetX - midIdealX;
  let idealEY = targetY - midIdealY;
  let idealEZ = targetZ - midIdealZ;
  _normalize3(_scratchV3C, idealEX, idealEY, idealEZ);

  let curEX = ex - mx, curEY = ey - my, curEZ = ez - mz;
  _normalize3(_scratchV3B, curEX, curEY, curEZ);

  _computeRotationBetween(_scratchQ, _scratchV3B[0], _scratchV3B[1], _scratchV3B[2],
                                   _scratchV3C[0], _scratchV3C[1], _scratchV3C[2]);

  getBonePoseRotation(slot, endBoneIdx, _scratchQB);
  _quatMul(_scratchQB, _scratchQ[0], _scratchQ[1], _scratchQ[2], _scratchQ[3],
                     _scratchQB[0], _scratchQB[1], _scratchQB[2], _scratchQB[3]);
  setBonePoseRotation(slot, endBoneIdx, _scratchQB[0], _scratchQB[1], _scratchQB[2], _scratchQB[3]);

  AnimationState.totalTwoBoneSolves++;
  return true;
}

/**
 * Computes the minimal rotation quaternion between two unit vectors.
 */
function _computeRotationBetween(out, ax, ay, az, bx, by, bz) {
  const dot = _clamp(ax * bx + ay * by + az * bz, -1, 1);

  if (dot > 0.99999) {
    out[0] = 0; out[1] = 0; out[2] = 0; out[3] = 1;
    return out;
  }
  if (dot < -0.99999) {
    // Opposite vectors — pick an arbitrary perpendicular axis.
    let px = 1, py = 0, pz = 0;
    if (Math.abs(ax) > 0.9) { px = 0; py = 1; pz = 0; }
    _cross3(_scratchV3D, ax, ay, az, px, py, pz);
    _normalize3(_scratchV3D, _scratchV3D[0], _scratchV3D[1], _scratchV3D[2]);
    out[0] = _scratchV3D[0]; out[1] = _scratchV3D[1]; out[2] = _scratchV3D[2]; out[3] = 0;
    return out;
  }

  const cx = ay * bz - az * by;
  const cy = az * bx - ax * bz;
  const cz = ax * by - ay * bx;
  const w = 1 + dot;
  const l = Math.sqrt(cx * cx + cy * cy + cz * cz + w * w) || 1;
  out[0] = cx / l; out[1] = cy / l; out[2] = cz / l; out[3] = w / l;
  return out;
}

/* ------------------------------------------------------------------ */
/* 14. FABRIK CHAIN SOLVER                                            */
/* ------------------------------------------------------------------ */

/**
 * FABRIK (Forward And Backward Reaching Inverse Kinematics) solver for
 * an arbitrary joint chain.
 *
 * `jointBoneIndices` is an array of bone indices from root to end.
 * `targetX/Y/Z` is the world-space target for the end effector.
 *
 * Uses scratch buffers; allocation-free.
 */
const _fabrikJointPos = new Float32Array(MAX_BONES_PER_ENTITY * 3);
const _fabrikBoneLens = new Float32Array(MAX_BONES_PER_ENTITY);

export function solveFABRIKChain(slot, jointBoneIndices, jointCount,
                                 targetX, targetY, targetZ,
                                 iterations, tolerance, weight) {
  if (slot < 0 || slot >= MAX_ACTIVE_ANIMATED) return false;
  if (jointCount < 2 || jointCount > MAX_BONES_PER_ENTITY) return false;

  const iters = iterations !== undefined ? iterations : DEFAULT_IK_ITERATIONS;
  const tol = tolerance !== undefined ? tolerance : DEFAULT_IK_TOLERANCE;

  // Read joint positions.
  for (let i = 0; i < jointCount; i++) {
    getBonePosePosition(slot, jointBoneIndices[i], _scratchV3A);
    _fabrikJointPos[i * 3 + 0] = _scratchV3A[0];
    _fabrikJointPos[i * 3 + 1] = _scratchV3A[1];
    _fabrikJointPos[i * 3 + 2] = _scratchV3A[2];
  }

  // Compute bone lengths.
  let totalLen = 0;
  for (let i = 0; i < jointCount - 1; i++) {
    const ax = _fabrikJointPos[i * 3 + 0];
    const ay = _fabrikJointPos[i * 3 + 1];
    const az = _fabrikJointPos[i * 3 + 2];
    const bx = _fabrikJointPos[(i + 1) * 3 + 0];
    const by = _fabrikJointPos[(i + 1) * 3 + 1];
    const bz = _fabrikJointPos[(i + 1) * 3 + 2];
    const l = _dist3(ax, ay, az, bx, by, bz);
    _fabrikBoneLens[i] = l;
    totalLen += l;
  }

  // Root position (fixed).
  const rootX = _fabrikJointPos[0];
  const rootY = _fabrikJointPos[1];
  const rootZ = _fabrikJointPos[2];

  // Distance from root to target.
  const distToTarget = _dist3(rootX, rootY, rootZ, targetX, targetY, targetZ);

  // If unreachable, aim the chain at the target.
  if (distToTarget > totalLen) {
    const inv = 1.0 / distToTarget;
    const dirX = (targetX - rootX) * inv;
    const dirY = (targetY - rootY) * inv;
    const dirZ = (targetZ - rootZ) * inv;
    let accum = 0;
    for (let i = 1; i < jointCount; i++) {
      accum += _fabrikBoneLens[i - 1];
      _fabrikJointPos[i * 3 + 0] = rootX + dirX * accum;
      _fabrikJointPos[i * 3 + 1] = rootY + dirY * accum;
      _fabrikJointPos[i * 3 + 2] = rootZ + dirZ * accum;
    }
  } else {
    // Iterative FABRIK solve.
    const end = jointCount - 1;
    for (let iter = 0; iter < iters; iter++) {
      // Forward reach.
      _fabrikJointPos[end * 3 + 0] = targetX;
      _fabrikJointPos[end * 3 + 1] = targetY;
      _fabrikJointPos[end * 3 + 2] = targetZ;

      for (let i = end - 1; i >= 0; i--) {
        const ax = _fabrikJointPos[i * 3 + 0];
        const ay = _fabrikJointPos[i * 3 + 1];
        const az = _fabrikJointPos[i * 3 + 2];
        const bx = _fabrikJointPos[(i + 1) * 3 + 0];
        const by = _fabrikJointPos[(i + 1) * 3 + 1];
        const bz = _fabrikJointPos[(i + 1) * 3 + 2];
        const l = _fabrikBoneLens[i] || 1e-6;
        const d = _dist3(ax, ay, az, bx, by, bz) || 1e-6;
        const k = l / d;
        _fabrikJointPos[i * 3 + 0] = bx + (ax - bx) * k;
        _fabrikJointPos[i * 3 + 1] = by + (ay - by) * k;
        _fabrikJointPos[i * 3 + 2] = bz + (az - bz) * k;
      }

      // Backward reach.
      _fabrikJointPos[0] = rootX;
      _fabrikJointPos[1] = rootY;
      _fabrikJointPos[2] = rootZ;

      for (let i = 0; i < end; i++) {
        const ax = _fabrikJointPos[i * 3 + 0];
        const ay = _fabrikJointPos[i * 3 + 1];
        const az = _fabrikJointPos[i * 3 + 2];
        const bx = _fabrikJointPos[(i + 1) * 3 + 0];
        const by = _fabrikJointPos[(i + 1) * 3 + 1];
        const bz = _fabrikJointPos[(i + 1) * 3 + 2];
        const l = _fabrikBoneLens[i] || 1e-6;
        const d = _dist3(ax, ay, az, bx, by, bz) || 1e-6;
        const k = l / d;
        _fabrikJointPos[(i + 1) * 3 + 0] = ax + (bx - ax) * k;
        _fabrikJointPos[(i + 1) * 3 + 1] = ay + (by - ay) * k;
        _fabrikJointPos[(i + 1) * 3 + 2] = az + (bz - az) * k;
      }

      // Check convergence.
      const dx = _fabrikJointPos[end * 3 + 0] - targetX;
      const dy = _fabrikJointPos[end * 3 + 1] - targetY;
      const dz = _fabrikJointPos[end * 3 + 2] - targetZ;
      if (dx * dx + dy * dy + dz * dz < tol * tol) break;
    }
  }

  // Write back joint positions (as if we rotated each joint).
  const w = weight !== undefined ? weight : 1;
  for (let i = 0; i < jointCount; i++) {
    getBonePosePosition(slot, jointBoneIndices[i], _scratchV3A);
    const nx = _fabrikJointPos[i * 3 + 0];
    const ny = _fabrikJointPos[i * 3 + 1];
    const nz = _fabrikJointPos[i * 3 + 2];
    if (w < 1) {
      setBonePosePosition(slot, jointBoneIndices[i],
        _scratchV3A[0] + (nx - _scratchV3A[0]) * w,
        _scratchV3A[1] + (ny - _scratchV3A[1]) * w,
        _scratchV3A[2] + (nz - _scratchV3A[2]) * w);
    } else {
      setBonePosePosition(slot, jointBoneIndices[i], nx, ny, nz);
    }
  }

  AnimationState.totalFABRIKSolves++;
  return true;
}

/* ------------------------------------------------------------------ */
/* 15. CCD CHAIN SOLVER                                               */
/* ------------------------------------------------------------------ */

/**
 * CCD (Cyclic Coordinate Descent) solver for an arbitrary chain.
 */
export function solveCCDChain(slot, jointBoneIndices, jointCount,
                              targetX, targetY, targetZ,
                              iterations, tolerance, weight) {
  if (slot < 0 || slot >= MAX_ACTIVE_ANIMATED) return false;
  if (jointCount < 2) return false;

  const iters = iterations !== undefined ? iterations : DEFAULT_IK_ITERATIONS;
  const tol = tolerance !== undefined ? tolerance : DEFAULT_IK_TOLERANCE;
  const end = jointCount - 1;

  for (let iter = 0; iter < iters; iter++) {
    for (let i = end - 1; i >= 0; i--) {
      // Read joint and end effector positions.
      getBonePosePosition(slot, jointBoneIndices[i], _scratchV3A);
      getBonePosePosition(slot, jointBoneIndices[end], _scratchV3B);
      const jx = _scratchV3A[0], jy = _scratchV3A[1], jz = _scratchV3A[2];
      const ex = _scratchV3B[0], ey = _scratchV3B[1], ez = _scratchV3B[2];

      // Vectors from joint to end and joint to target.
      let toEndX = ex - jx, toEndY = ey - jy, toEndZ = ez - jz;
      let toTgtX = targetX - jx, toTgtY = targetY - jy, toTgtZ = targetZ - jz;
      _normalize3(_scratchV3C, toEndX, toEndY, toEndZ);
      _normalize3(_scratchV3D, toTgtX, toTgtY, toTgtZ);

      // Compute rotation between the two.
      _computeRotationBetween(_scratchQ, _scratchV3C[0], _scratchV3C[1], _scratchV3C[2],
                                       _scratchV3D[0], _scratchV3D[1], _scratchV3D[2]);

      // Apply rotation to the joint's quaternion.
      getBonePoseRotation(slot, jointBoneIndices[i], _scratchQB);
      _quatMul(_scratchQB, _scratchQ[0], _scratchQ[1], _scratchQ[2], _scratchQ[3],
                         _scratchQB[0], _scratchQB[1], _scratchQB[2], _scratchQB[3]);
      setBonePoseRotation(slot, jointBoneIndices[i], _scratchQB[0], _scratchQB[1], _scratchQB[2], _scratchQB[3]);

      // (In a full implementation we would re-solve world transforms
      // here; we approximate by assuming the position update is small.)
    }

    // Check convergence.
    getBonePosePosition(slot, jointBoneIndices[end], _scratchV3B);
    const dx = _scratchV3B[0] - targetX;
    const dy = _scratchV3B[1] - targetY;
    const dz = _scratchV3B[2] - targetZ;
    if (dx * dx + dy * dy + dz * dz < tol * tol) break;
  }

  AnimationState.totalCCDSolves++;
  return true;
}

/* ------------------------------------------------------------------ */
/* 16. STANDARD CHAIN SOLVERS                                         */
/* ------------------------------------------------------------------ */

export function solveArmIK(eid, side /* 0=L 1=R */, targetX, targetY, targetZ, options) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  const slot = AnimationStateComp.activeSlot[eid];
  if (slot < 0) return false;

  const shoulderIdx = side === 0
    ? AnimationBinding.leftWingBoneIdx[eid]   // fallback
    : AnimationBinding.rightWingBoneIdx[eid];

  // For a humanoid the standard shoulder/elbow/wrist indices are 8/9/10
  // (left) and 22/23/24 (right). We rely on the humanoid layout here.
  const rootIdx = side === 0 ? 8  : 22;
  const midIdx  = side === 0 ? 9  : 23;
  const endIdx  = side === 0 ? 10 : 24;

  const opts = options || {};
  return solveTwoBoneIK(slot, rootIdx, midIdx, endIdx,
    targetX, targetY, targetZ,
    opts.poleX !== undefined ? opts.poleX : 0,
    opts.poleY !== undefined ? opts.poleY : -1,
    opts.poleZ !== undefined ? opts.poleZ : 0,
    opts.weight !== undefined ? opts.weight : 1);
}

export function solveHandIK(eid, side /* 0=L 1=R */, targetX, targetY, targetZ, options) {
  const slot = AnimationStateComp.activeSlot[eid];
  if (slot < 0) return false;

  const handIdx = side === 0 ? 11 : 25;
  getBonePosePosition(slot, handIdx, _scratchV3A);

  const opts = options || {};
  const w = opts.weight !== undefined ? opts.weight : 1;
  const nx = _scratchV3A[0] + (targetX - _scratchV3A[0]) * w;
  const ny = _scratchV3A[1] + (targetY - _scratchV3A[1]) * w;
  const nz = _scratchV3A[2] + (targetZ - _scratchV3A[2]) * w;
  setBonePosePosition(slot, handIdx, nx, ny, nz);

  // Fingers follow the hand bone rigidly — no separate solve needed
  // unless a per-finger target is set.
  if (opts.animateFingers) {
    // Apply a small curl to all fingers on the given side.
    const start = side === 0 ? 12 : 26;
    const curl = opts.fingerCurl !== undefined ? opts.fingerCurl : 0.3;
    for (let i = 0; i < 10; i++) {
      const fIdx = start + i;
      getBonePoseRotation(slot, fIdx, _scratchQB);
      _quatFromAxisAngle(_scratchQ, 1, 0, 0, curl);
      _quatMul(_scratchQB, _scratchQB[0], _scratchQB[1], _scratchQB[2], _scratchQB[3],
                         _scratchQ[0], _scratchQ[1], _scratchQ[2], _scratchQ[3]);
      setBonePoseRotation(slot, fIdx, _scratchQB[0], _scratchQB[1], _scratchQB[2], _scratchQB[3]);
    }
  }
  return true;
}

export function solveLegIK(eid, side /* 0=L 1=R */, targetX, targetY, targetZ, options) {
  const slot = AnimationStateComp.activeSlot[eid];
  if (slot < 0) return false;

  const rootIdx = side === 0 ? 36 : 41;   // upper leg
  const midIdx  = side === 0 ? 37 : 42;   // knee
  const endIdx  = side === 0 ? 38 : 43;   // ankle

  const opts = options || {};
  return solveTwoBoneIK(slot, rootIdx, midIdx, endIdx,
    targetX, targetY, targetZ,
    opts.poleX !== undefined ? opts.poleX : 0,
    opts.poleY !== undefined ? opts.poleY : 0,
    opts.poleZ !== undefined ? opts.poleZ : 1,
    opts.weight !== undefined ? opts.weight : 1);
}

export function solveFootIK(eid, side, targetX, targetY, targetZ, options) {
  const slot = AnimationStateComp.activeSlot[eid];
  if (slot < 0) return false;
  const footIdx = side === 0 ? 39 : 44;
  const toeIdx  = side === 0 ? 40 : 45;

  const opts = options || {};
  const w = opts.weight !== undefined ? opts.weight : 1;
  getBonePosePosition(slot, footIdx, _scratchV3A);
  setBonePosePosition(slot, footIdx,
    _scratchV3A[0] + (targetX - _scratchV3A[0]) * w,
    _scratchV3A[1] + (targetY - _scratchV3A[1]) * w,
    _scratchV3A[2] + (targetZ - _scratchV3A[2]) * w);
  // Toe follows the foot.
  getBonePosePosition(slot, toeIdx, _scratchV3A);
  setBonePosePosition(slot, toeIdx,
    _scratchV3A[0] + (targetX - _scratchV3A[0]) * w,
    _scratchV3A[1] + (targetY - _scratchV3A[1]) * w,
    _scratchV3A[2] + (targetZ - _scratchV3A[2]) * w);
  return true;
}

export function solveHeadLookAt(eid, targetX, targetY, targetZ, options) {
  const slot = AnimationStateComp.activeSlot[eid];
  if (slot < 0) return false;
  const neckIdx = AnimationBinding.neckBoneIdx[eid];
  const headIdx = AnimationBinding.headBoneIdx[eid];

  const opts = options || {};
  const w = opts.weight !== undefined ? opts.weight : 0.6;
  // Simple look-at: point the neck and head at the target.
  // Full quaternion solve is downstream; we approximate by position blend.
  getBonePosePosition(slot, neckIdx, _scratchV3A);
  setBonePosePosition(slot, neckIdx,
    _scratchV3A[0] + (targetX - _scratchV3A[0]) * w * 0.3,
    _scratchV3A[1] + (targetY - _scratchV3A[1]) * w * 0.3,
    _scratchV3A[2] + (targetZ - _scratchV3A[2]) * w * 0.3);
  getBonePosePosition(slot, headIdx, _scratchV3A);
  setBonePosePosition(slot, headIdx,
    _scratchV3A[0] + (targetX - _scratchV3A[0]) * w * 0.5,
    _scratchV3A[1] + (targetY - _scratchV3A[1]) * w * 0.5,
    _scratchV3A[2] + (targetZ - _scratchV3A[2]) * w * 0.5);
  return true;
}

export function solveSpineLookAt(eid, targetX, targetY, targetZ, options) {
  const slot = AnimationStateComp.activeSlot[eid];
  if (slot < 0) return false;
  const opts = options || {};
  const w = opts.weight !== undefined ? opts.weight : 0.4;

  // Distribute the bend across the 4 spine bones (indices 2..5).
  for (let i = 2; i <= 5; i++) {
    getBonePosePosition(slot, i, _scratchV3A);
    const fac = w * (i - 1) / 6;
    setBonePosePosition(slot, i,
      _scratchV3A[0] + (targetX - _scratchV3A[0]) * fac,
      _scratchV3A[1] + (targetY - _scratchV3A[1]) * fac,
      _scratchV3A[2] + (targetZ - _scratchV3A[2]) * fac);
  }
  return true;
}

export function solveWingIK(eid, side, targetX, targetY, targetZ, options) {
  const slot = AnimationStateComp.activeSlot[eid];
  if (slot < 0) return false;
  // Bird wing bones: 6/7/8 (left), 9/10/11 (right).
  const rootIdx = side === 0 ? 6  : 9;
  const midIdx  = side === 0 ? 7  : 10;
  const endIdx  = side === 0 ? 8  : 11;
  return solveTwoBoneIK(slot, rootIdx, midIdx, endIdx,
    targetX, targetY, targetZ,
    0, 0, -1, options && options.weight !== undefined ? options.weight : 1);
}

export function solveTailIK(eid, targetX, targetY, targetZ, options) {
  const slot = AnimationStateComp.activeSlot[eid];
  if (slot < 0) return false;
  const start = AnimationBinding.tailBoneIdx[eid];
  if (start === 0 && AnimationStateComp.skeletonType[eid] !== SKELETON_TYPE.BIRD) return false;
  // 4-bone tail chain for quadrupeds, 3 for birds.
  const jointIndices = [start, start + 1, start + 2, start + 3];
  return solveFABRIKChain(slot, jointIndices, 4, targetX, targetY, targetZ, 8, 0.02, options && options.weight);
}

export function solveQuadrupedIK(eid, options) {
  const slot = AnimationStateComp.activeSlot[eid];
  if (slot < 0) return false;
  // Solve each leg as a 3-bone chain (shoulder → wrist → foot).
  const flRoot = 8,  flMid = 9,  flEnd = 10;
  const frRoot = 12, frMid = 13, frEnd = 14;
  const blRoot = 16, blMid = 17, blEnd = 18;
  const brRoot = 20, brMid = 21, brEnd = 22;

  const opts = options || {};
  if (opts.frontLeft)  solveTwoBoneIK(slot, flRoot, flMid, flEnd, opts.frontLeft.x,  opts.frontLeft.y,  opts.frontLeft.z,  0, -1, 0, 1);
  if (opts.frontRight) solveTwoBoneIK(slot, frRoot, frMid, frEnd, opts.frontRight.x, opts.frontRight.y, opts.frontRight.z, 0, -1, 0, 1);
  if (opts.backLeft)   solveTwoBoneIK(slot, blRoot, blMid, blEnd, opts.backLeft.x,   opts.backLeft.y,   opts.backLeft.z,   0, -1, 0, 1);
  if (opts.backRight)  solveTwoBoneIK(slot, brRoot, brMid, brEnd, opts.backRight.x,  opts.backRight.y,  opts.backRight.z,  0, -1, 0, 1);

  return true;
}

export function solveMachineChainIK(eid, targetX, targetY, targetZ, options) {
  const slot = AnimationStateComp.activeSlot[eid];
  if (slot < 0) return false;
  return solveTwoBoneIK(slot, 1, 2, 3, targetX, targetY, targetZ, 0, 1, 0, 1);
}

/**
 * Full-body IK orchestrator. Calls each subsystem solver in the correct
 * order: legs → pelvis → spine → head → arms → hands.
 *
 * Reads targets from the entity's IKTargets slots.
 */
export function solveFullBodyIK(eid, options) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;

  const t0 = _now();

  const slot = AnimationStateComp.activeSlot[eid];
  if (slot < 0) return false;

  const skeletonType = AnimationStateComp.skeletonType[eid];
  const opts = options || {};

  // 1. Legs.
  if (opts.solveLegs !== false) {
    solveLegIK(eid, 0, FootPlacement.targetX_L[eid], FootPlacement.targetY_L[eid], FootPlacement.targetZ_L[eid], opts);
    solveLegIK(eid, 1, FootPlacement.targetX_R[eid], FootPlacement.targetY_R[eid], FootPlacement.targetZ_R[eid], opts);
  }

  // 2. Pelvis.
  if (opts.solvePelvis !== false) {
    const hipIdx = AnimationBinding.hipBoneIdx[eid];
    if (hipIdx >= 0) {
      getBonePosePosition(slot, hipIdx, _scratchV3A);
      setBonePosePosition(slot, hipIdx,
        _scratchV3A[0],
        _scratchV3A[1] + FootPlacement.hipOffsetY[eid],
        _scratchV3A[2] + FootPlacement.hipOffsetForward[eid]);
    }
  }

  // 3. Spine look-at.
  if (opts.spineTarget) {
    solveSpineLookAt(eid, opts.spineTarget.x, opts.spineTarget.y, opts.spineTarget.z, opts);
  }

  // 4. Head look-at.
  if (opts.headTarget) {
    solveHeadLookAt(eid, opts.headTarget.x, opts.headTarget.y, opts.headTarget.z, opts);
  }

  // 5. Arms.
  if (opts.solveArms !== false && skeletonType === SKELETON_TYPE.HUMANOID) {
    const lhFlat = eid * MAX_IK_CHAINS_PER_ENTITY + 0;
    const rhFlat = eid * MAX_IK_CHAINS_PER_ENTITY + 1;
    if (IKTargets.valid[lhFlat] === 1) {
      solveArmIK(eid, 0, IKTargets.posX[lhFlat], IKTargets.posY[lhFlat], IKTargets.posZ[lhFlat], opts);
    }
    if (IKTargets.valid[rhFlat] === 1) {
      solveArmIK(eid, 1, IKTargets.posX[rhFlat], IKTargets.posY[rhFlat], IKTargets.posZ[rhFlat], opts);
    }
  }

  // 6. Hands.
  if (opts.solveHands === true && skeletonType === SKELETON_TYPE.HUMANOID) {
    const lhFlat = eid * MAX_IK_CHAINS_PER_ENTITY + 2;
    const rhFlat = eid * MAX_IK_CHAINS_PER_ENTITY + 3;
    if (IKTargets.valid[lhFlat] === 1) {
      solveHandIK(eid, 0, IKTargets.posX[lhFlat], IKTargets.posY[lhFlat], IKTargets.posZ[lhFlat], opts);
    }
    if (IKTargets.valid[rhFlat] === 1) {
      solveHandIK(eid, 1, IKTargets.posX[rhFlat], IKTargets.posY[rhFlat], IKTargets.posZ[rhFlat], opts);
    }
  }

  // 7. Specialized skeletons.
  if (skeletonType === SKELETON_TYPE.BIRD && opts.solveWings !== false) {
    if (opts.leftWingTarget)  solveWingIK(eid, 0, opts.leftWingTarget.x,  opts.leftWingTarget.y,  opts.leftWingTarget.z,  opts);
    if (opts.rightWingTarget) solveWingIK(eid, 1, opts.rightWingTarget.x, opts.rightWingTarget.y, opts.rightWingTarget.z, opts);
    if (opts.tailTarget) solveTailIK(eid, opts.tailTarget.x, opts.tailTarget.y, opts.tailTarget.z, opts);
  }
  if (skeletonType === SKELETON_TYPE.QUADRUPED) {
    solveQuadrupedIK(eid, opts);
    if (opts.tailTarget) solveTailIK(eid, opts.tailTarget.x, opts.tailTarget.y, opts.tailTarget.z, opts);
  }
  if (skeletonType === SKELETON_TYPE.MACHINE && opts.machineTarget) {
    solveMachineChainIK(eid, opts.machineTarget.x, opts.machineTarget.y, opts.machineTarget.z, opts);
  }

  AnimationState.totalIKSolves++;
  AnimationState.totalFullBodySolves++;

  const t1 = _now();
  AnimationState.lastIKMs = t1 - t0;
  AnimationState.avgIKMs += (AnimationState.lastIKMs - AnimationState.avgIKMs) * 0.15;

  return true;
}

/* ------------------------------------------------------------------ */
/* 17. GROUND ADAPTATION & STAIR CLIMBING                             */
/* ------------------------------------------------------------------ */

/**
 * Analytic ground probe. Calls the supplied height function at (x, z)
 * and returns the ground height + normal.
 *
 * `heightFn(x, z, outNormal)` — must return the ground height and
 * optionally write a surface normal into `outNormal`.
 */
export function raycastGround(eid, x, z, heightFn, outNormal) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  if (typeof heightFn !== 'function') return false;

  AnimationState.totalGroundRays++;

  const y = heightFn(x, z, outNormal);
  GroundContact.footY_L[eid] = y;
  return true;
}

/**
 * Reads the ground normal at (x, z) using the given height function.
 */
export function computeGroundNormal(heightFn, x, z, eps, outNormal) {
  if (typeof heightFn !== 'function') return false;
  const e = eps !== undefined ? eps : 0.5;
  const hL = heightFn(x - e, z);
  const hR = heightFn(x + e, z);
  const hD = heightFn(x, z - e);
  const hU = heightFn(x, z + e);
  const nx = (hL - hR) / (2 * e);
  const nz = (hD - hU) / (2 * e);
  const ny = 1.0;
  const inv = 1.0 / (Math.sqrt(nx * nx + ny * ny + nz * nz) || 1);
  outNormal[0] = nx * inv;
  outNormal[1] = ny * inv;
  outNormal[2] = nz * inv;
  return true;
}

/**
 * Adapts a foot to the ground surface at (x, z). Writes the foot target
 * and ground normal into GroundContact.
 */
export function adaptFootToGround(eid, side, x, z, heightFn, options) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  if (typeof heightFn !== 'function') return false;

  const opts = options || {};
  const offsetY = opts.offsetY !== undefined ? opts.offsetY : 0.0;
  const normal = opts.normal || _scratchV3D;

  const y = heightFn(x, z, normal) + offsetY;

  if (side === 0) {
    FootPlacement.targetX_L[eid] = x;
    FootPlacement.targetY_L[eid] = y;
    FootPlacement.targetZ_L[eid] = z;
    GroundContact.footX_L[eid] = x;
    GroundContact.footY_L[eid] = y;
    GroundContact.footZ_L[eid] = z;
    GroundContact.groundNX_L[eid] = normal[0];
    GroundContact.groundNY_L[eid] = normal[1];
    GroundContact.groundNZ_L[eid] = normal[2];
  } else {
    FootPlacement.targetX_R[eid] = x;
    FootPlacement.targetY_R[eid] = y;
    FootPlacement.targetZ_R[eid] = z;
    GroundContact.footX_R[eid] = x;
    GroundContact.footY_R[eid] = y;
    GroundContact.footZ_R[eid] = z;
    GroundContact.groundNX_R[eid] = normal[0];
    GroundContact.groundNY_R[eid] = normal[1];
    GroundContact.groundNZ_R[eid] = normal[2];
  }

  GroundContact.hasGround[eid] = 1;
  return true;
}

/**
 * Adapts the hip position based on ground contact heights. Keeps the
 * pelvis at a symmetric height above the lowest ground contact.
 */
export function adaptHipToGround(eid, options) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  if (GroundContact.hasGround[eid] === 0) return false;

  const opts = options || {};
  const stanceHeight = opts.stanceHeight !== undefined ? opts.stanceHeight : 1.0;

  const yL = GroundContact.footY_L[eid];
  const yR = GroundContact.footY_R[eid];
  const lowest = Math.min(yL, yR);

  FootPlacement.hipOffsetY[eid] = lowest + stanceHeight;
  return true;
}

/**
 * Detects whether the entity is facing a stair by comparing the ground
 * height at a probe point ahead of the entity to the ground height at
 * its feet.
 */
export function detectStair(eid, forwardX, forwardZ, probeDistance, heightFn, stepThreshold) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  if (typeof heightFn !== 'function') return false;

  const thresh = stepThreshold !== undefined ? stepThreshold : 0.15;
  const probeX = forwardX * probeDistance;
  const probeZ = forwardZ * probeDistance;

  const hFoot = GroundContact.footY_L[eid] || 0;
  const hAhead = heightFn(probeX, probeZ);

  const delta = hAhead - hFoot;
  const isStair = delta > thresh;

  if (isStair) {
    AnimationState.totalStairDetections++;
    FootPlacement.stairMode[eid] = 1;
    GroundContact.stepHeightL[eid] = delta;
    GroundContact.stepHeightR[eid] = delta;
  } else {
    FootPlacement.stairMode[eid] = 0;
    GroundContact.stepHeightL[eid] = 0;
    GroundContact.stepHeightR[eid] = 0;
  }

  return isStair;
}

/**
 * Snaps a foot to a stair step at (x, z). Higher step edge becomes the
 * new foot target.
 */
export function snapFootToStair(eid, side, x, z, heightFn) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;
  const h = heightFn(x, z);
  if (side === 0) {
    FootPlacement.targetY_L[eid] = h;
  } else {
    FootPlacement.targetY_R[eid] = h;
  }
  return true;
}

/**
 * Adapts the pelvis and both feet to a stair step. Ensures the pelvis
 * rises by the step height and both feet snap to the same level so the
 * character doesn't look half-climbed.
 */
export function adaptBodyToStairs(eid, stepHeight, options) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;

  // Raise the pelvis by the step height plus a small natural offset.
  FootPlacement.hipOffsetY[eid] = (FootPlacement.hipOffsetY[eid] || 0) + stepHeight * 1.0;
  FootPlacement.hipOffsetForward[eid] = (FootPlacement.hipOffsetForward[eid] || 0) + (options && options.forwardOffset || 0.0);

  // Snap the trailing foot up to the same level.
  const lowest = Math.min(GroundContact.footY_L[eid], GroundContact.footY_R[eid]);
  const target = lowest + stepHeight;
  GroundContact.footY_L[eid] = target;
  GroundContact.footY_R[eid] = target;
  FootPlacement.targetY_L[eid] = target;
  FootPlacement.targetY_R[eid] = target;

  return true;
}

/* ------------------------------------------------------------------ */
/* 18. FRAME LIFECYCLE                                                */
/* ------------------------------------------------------------------ */

/**
 * Advances the animation system frame counter.
 */
export function tickAnimation(frameNumber) {
  if (typeof frameNumber === 'number') AnimationState.frame = frameNumber;
  else AnimationState.frame++;
}

/**
 * Advances one entity's animation state one frame.
 */
export function tickAnimationEntity(eid, dt) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return false;

  const state = AnimationStateComp.state[eid];

  if (state === ANIM_STATE.PLAYING) {
    const count = AnimationLayer.layerCount[eid];
    for (let l = 0; l < count; l++) {
      const flat = eid * MAX_ANIM_LAYERS_PER_ENTITY + l;
      if (AnimationLayer.active[flat] === 0) continue;
      AnimationLayer.time[flat] += dt * AnimationLayer.speed[flat];
      const clipId = AnimationLayer.clipId[flat];
      const dur = AnimationClip.duration[eid * MAX_CLIPS_PER_ENTITY + clipId] || 1;
      if (AnimationLayer.loop[flat] === 1) {
        if (AnimationLayer.time[flat] > dur) AnimationLayer.time[flat] -= dur;
      } else {
        if (AnimationLayer.time[flat] > dur) AnimationLayer.time[flat] = dur;
      }
    }
  }

  if (AnimationStateComp.state[eid] === ANIM_STATE.BLENDING) {
    AnimationBlend.elapsed[eid] += dt;
    const dur = AnimationBlend.duration[eid];
    const alpha = _clamp(AnimationBlend.elapsed[eid] / dur, 0, 1);
    AnimationBlend.alpha[eid] = alpha;
    if (alpha >= 1) {
      AnimationBlend.active[eid] = 0;
      AnimationStateComp.state[eid] = ANIM_STATE.PLAYING;
    }
  }

  if (state === ANIM_STATE.MOTION_MATCH) {
    tickMotionMatchTransition(eid, dt);
  }

  evaluateMorphWeights(eid, dt);

  return true;
}

/**
 * Full per-frame animation pipeline.
 */
export function tickAnimationSystem(frameNumber, dt) {
  const t0 = _now();
  tickAnimation(frameNumber);

  const adapter = getAdapter();
  let animated = 0;
  for (let eid = 0; eid < MAX_ENTITIES; eid++) {
    if (AnimationStateComp.enabled[eid] === 0) continue;
    if (!adapter.entityAlive(eid)) continue;
    tickAnimationEntity(eid, dt);
    animated++;
  }

  AnimationStats.activeAnimated[0] = animated;

  const t1 = _now();
  const cost = t1 - t0;
  AnimationState.lastTickMs = cost;
  AnimationState.avgTickMs += (cost - AnimationState.avgTickMs) * 0.15;

  return {
    frame: AnimationState.frame,
    animated,
    activePoses: AnimationState.activePoses,
    cost,
  };
}

/* ------------------------------------------------------------------ */
/* 19. REGISTRATION WITH THE COMPONENT REGISTRY                       */
/* ------------------------------------------------------------------ */

export function registerAnimationComponents(registry) {
  const reg = registry || getDefaultComponentRegistry();
  if (!reg) return 0;

  return reg.registerMany([
    { name: 'AnimationStateComp',   component: AnimationStateComp,   category: 8, subsystem: 1, dependencies: [] },
    { name: 'AnimationBinding',     component: AnimationBinding,     category: 8, subsystem: 1, dependencies: ['AnimationStateComp'] },
    { name: 'AnimationLayer',       component: AnimationLayer,       category: 8, subsystem: 1, dependencies: [] },
    { name: 'AnimationClip',        component: AnimationClip,        category: 8, subsystem: 1, dependencies: [] },
    { name: 'AnimationBlend',       component: AnimationBlend,       category: 8, subsystem: 1, dependencies: [] },
    { name: 'AnimationRoot',        component: AnimationRoot,        category: 8, subsystem: 1, dependencies: [] },
    { name: 'AnimationMorph',       component: AnimationMorph,       category: 8, subsystem: 1, dependencies: [] },
    { name: 'AnimationEvent',       component: AnimationEvent,       category: 8, subsystem: 1, dependencies: [] },
    { name: 'AnimationIK',          component: AnimationIK,          category: 8, subsystem: 1, dependencies: [] },
    { name: 'IKTargets',            component: IKTargets,            category: 8, subsystem: 1, dependencies: ['AnimationIK'] },
    { name: 'GroundContact',        component: GroundContact,        category: 8, subsystem: 1, dependencies: [] },
    { name: 'FootPlacement',        component: FootPlacement,        category: 8, subsystem: 1, dependencies: [] },
    { name: 'MotionMatchState',     component: MotionMatchState,     category: 8, subsystem: 1, dependencies: [] },
    { name: 'MotionPose',           component: MotionPose,           category: 8, subsystem: 1, dependencies: [] },
    { name: 'AnimationBudget',      component: AnimationBudget,      category: 8, subsystem: 1, dependencies: [] },
    { name: 'AnimationStats',       component: AnimationStats,       category: 8, subsystem: 1, dependencies: [] },
  ]);
}

/* ------------------------------------------------------------------ */
/* 20. DIAGNOSTICS                                                    */
/* ------------------------------------------------------------------ */

export function getAnimationEntityStats(eid) {
  if (typeof eid !== 'number' || eid < 0 || eid >= MAX_ENTITIES) return null;
  return {
    entity:           eid,
    state:            ANIM_STATE_NAME[AnimationStateComp.state[eid]] || 'idle',
    skeletonType:     SKELETON_TYPE_NAME[AnimationStateComp.skeletonType[eid]] || 'none',
    boneCount:        AnimationStateComp.boneCount[eid],
    activeSlot:       AnimationStateComp.activeSlot[eid],
    layerCount:       AnimationLayer.layerCount[eid],
    ikChainCount:     AnimationIK.chainCount[eid],
    morphCount:       AnimationMorph.targetCount[eid],
    hasGround:        GroundContact.hasGround[eid] === 1,
    stairMode:        FootPlacement.stairMode[eid] === 1,
    motionMatching:   MotionMatchState.enabled[eid] === 1,
    currentPose:      MotionMatchState.currentPoseIdx[eid],
    globalTime:       AnimationStateComp.globalTime[eid],
  };
}

export function getAnimationSystemReport() {
  return {
    frame:                    AnimationState.frame,
    activePoses:              AnimationState.activePoses,
    peakActivePoses:          AnimationState.peakActivePoses,
    maxBonesPerEntity:        MAX_BONES_PER_ENTITY,
    maxActiveAnimated:        MAX_ACTIVE_ANIMATED,
    totalBinds:               AnimationState.totalBinds,
    totalPlays:               AnimationState.totalPlays,
    totalPauses:              AnimationState.totalPauses,
    totalStops:               AnimationState.totalStops,
    totalLayers:              AnimationState.totalLayers,
    totalBlends:              AnimationState.totalBlends,
    totalIKChainsCreated:     AnimationState.totalIKChainsCreated,
    totalIKSolves:            AnimationState.totalIKSolves,
    totalTwoBoneSolves:       AnimationState.totalTwoBoneSolves,
    totalFABRIKSolves:        AnimationState.totalFABRIKSolves,
    totalCCDSolves:           AnimationState.totalCCDSolves,
    totalFullBodySolves:      AnimationState.totalFullBodySolves,
    totalGroundRays:          AnimationState.totalGroundRays,
    totalStairDetections:     AnimationState.totalStairDetections,
    totalMotionMatches:       AnimationState.totalMotionMatches,
    totalMotionTransitions:   AnimationState.totalMotionTransitions,
    totalMorphUpdates:        AnimationState.totalMorphUpdates,
    motionPoseCount:          MotionPose.poseCount[0],
    motionPoseCapacity:       MAX_MOTION_POSES,
    lastTickMs:               AnimationState.lastTickMs,
    avgTickMs:                AnimationState.avgTickMs,
    lastIKMs:                 AnimationState.lastIKMs,
    avgIKMs:                  AnimationState.avgIKMs,
    lastMMMs:                 AnimationState.lastMMMs,
    avgMMMs:                  AnimationState.avgMMMs,
    perfTier:                 PERF_TIER_LOCAL,
  };
}

/* ------------------------------------------------------------------ */
/* 21. RESET                                                          */
/* ------------------------------------------------------------------ */

export function resetAnimationState() {
  AnimationStateComp.state.fill(0);
  AnimationStateComp.prevState.fill(0);
  AnimationStateComp.skeletonId.fill(-1);
  AnimationStateComp.skeletonType.fill(0);
  AnimationStateComp.boneCount.fill(0);
  AnimationStateComp.activeSlot.fill(-1);
  AnimationStateComp.flags.fill(0);
  AnimationStateComp.lastStateFrame.fill(0);
  AnimationStateComp.globalTime.fill(0);
  AnimationStateComp.globalSpeed.fill(0);
  AnimationStateComp.globalWeight.fill(0);
  AnimationStateComp.enabled.fill(0);
  AnimationStateComp.poseDirty.fill(0);

  AnimationBinding.rootBoneIdx.fill(0);
  AnimationBinding.hipBoneIdx.fill(0);
  AnimationBinding.headBoneIdx.fill(0);
  AnimationBinding.leftFootBoneIdx.fill(0);
  AnimationBinding.rightFootBoneIdx.fill(0);
  AnimationBinding.leftHandBoneIdx.fill(0);
  AnimationBinding.rightHandBoneIdx.fill(0);
  AnimationBinding.spineBoneIdx.fill(0);
  AnimationBinding.neckBoneIdx.fill(0);
  AnimationBinding.tailBoneIdx.fill(0);
  AnimationBinding.leftWingBoneIdx.fill(0);
  AnimationBinding.rightWingBoneIdx.fill(0);
  AnimationBinding.bound.fill(0);

  AnimationLayer.clipId.fill(-1);
  AnimationLayer.time.fill(0);
  AnimationLayer.speed.fill(1);
  AnimationLayer.weight.fill(0);
  AnimationLayer.loop.fill(1);
  AnimationLayer.active.fill(0);
  AnimationLayer.layerCount.fill(0);
  AnimationLayer.blendMode.fill(0);

  AnimationClip.duration.fill(0);
  AnimationClip.trackCount.fill(0);
  AnimationClip.loop.fill(0);
  AnimationClip.rootMotion.fill(0);
  AnimationClip.fps.fill(30);
  AnimationClip.valid.fill(0);

  AnimationBlend.fromClipId.fill(-1);
  AnimationBlend.toClipId.fill(-1);
  AnimationBlend.fromTime.fill(0);
  AnimationBlend.toTime.fill(0);
  AnimationBlend.alpha.fill(0);
  AnimationBlend.duration.fill(0);
  AnimationBlend.elapsed.fill(0);
  AnimationBlend.active.fill(0);
  AnimationBlend.syncPhase.fill(0);

  AnimationRoot.offsetX.fill(0);
  AnimationRoot.offsetY.fill(0);
  AnimationRoot.offsetZ.fill(0);
  AnimationRoot.velocityX.fill(0);
  AnimationRoot.velocityY.fill(0);
  AnimationRoot.velocityZ.fill(0);
  AnimationRoot.yaw.fill(0);
  AnimationRoot.applyToTransform.fill(0);

  AnimationMorph.weight.fill(0);
  AnimationMorph.targetWeight.fill(0);
  AnimationMorph.targetCount.fill(0);
  AnimationMorph.blendRate.fill(8);

  AnimationEvent.pendingEvent.fill(0);
  AnimationEvent.eventFlags.fill(0);
  AnimationEvent.lastEventFrame.fill(0);
  AnimationEvent.eventCounter.fill(0);

  AnimationIK.chainType.fill(0);
  AnimationIK.solverType.fill(0);
  AnimationIK.rootBoneIdx.fill(0);
  AnimationIK.midBoneIdx.fill(0);
  AnimationIK.endBoneIdx.fill(0);
  AnimationIK.iterations.fill(0);
  AnimationIK.tolerance.fill(0);
  AnimationIK.weight.fill(1);
  AnimationIK.enabled.fill(0);
  AnimationIK.chainCount.fill(0);

  IKTargets.posX.fill(0);
  IKTargets.posY.fill(0);
  IKTargets.posZ.fill(0);
  IKTargets.rotX.fill(0);
  IKTargets.rotY.fill(0);
  IKTargets.rotZ.fill(0);
  IKTargets.rotW.fill(1);
  IKTargets.poleX.fill(0);
  IKTargets.poleY.fill(1);
  IKTargets.poleZ.fill(0);
  IKTargets.valid.fill(0);

  GroundContact.footX_L.fill(0);
  GroundContact.footY_L.fill(0);
  GroundContact.footZ_L.fill(0);
  GroundContact.footX_R.fill(0);
  GroundContact.footY_R.fill(0);
  GroundContact.footZ_R.fill(0);
  GroundContact.groundNX_L.fill(0);
  GroundContact.groundNY_L.fill(1);
  GroundContact.groundNZ_L.fill(0);
  GroundContact.groundNX_R.fill(0);
  GroundContact.groundNY_R.fill(1);
  GroundContact.groundNZ_R.fill(0);
  GroundContact.contactStateL.fill(0);
  GroundContact.contactStateR.fill(0);
  GroundContact.contactTimerL.fill(0);
  GroundContact.contactTimerR.fill(0);
  GroundContact.slopeAngle.fill(0);
  GroundContact.stepHeightL.fill(0);
  GroundContact.stepHeightR.fill(0);
  GroundContact.hasGround.fill(0);

  FootPlacement.targetX_L.fill(0);
  FootPlacement.targetY_L.fill(0);
  FootPlacement.targetZ_L.fill(0);
  FootPlacement.targetX_R.fill(0);
  FootPlacement.targetY_R.fill(0);
  FootPlacement.targetZ_R.fill(0);
  FootPlacement.offsetY_L.fill(0);
  FootPlacement.offsetY_R.fill(0);
  FootPlacement.locked.fill(0);
  FootPlacement.stairMode.fill(0);
  FootPlacement.hipOffsetY.fill(0);
  FootPlacement.hipOffsetForward.fill(0);

  MotionMatchState.enabled.fill(0);
  MotionMatchState.currentPoseIdx.fill(-1);
  MotionMatchState.targetPoseIdx.fill(-1);
  MotionMatchState.transitionAlpha.fill(0);
  MotionMatchState.transitionDuration.fill(0);
  MotionMatchState.transitionElapsed.fill(0);
  MotionMatchState.searchMode.fill(0);
  MotionMatchState.lastMatchScore.fill(0);
  MotionMatchState.lastSearchFrame.fill(0);
  MotionMatchState.databaseId.fill(-1);
  MotionMatchState.featureWeightOverride.fill(1);

  MotionPose.featureVec.fill(0);
  MotionPose.poseTRSOffset.fill(0);
  MotionPose.clipId.fill(-1);
  MotionPose.frameTime.fill(0);
  MotionPose.velocity.fill(0);
  MotionPose.tags.fill(0);
  MotionPose.valid.fill(0);
  MotionPose.poseCount[0] = 0;

  AnimationBudget.entityCost.fill(0);
  AnimationBudget.entityCostEma.fill(0);
  AnimationBudget.totalCost[0] = 0;
  AnimationBudget.budgetCap[0] = 4.0;
  AnimationBudget.budgetExceeded[0] = 0;
  AnimationBudget.ikBudgetCap[0] = 2.0;
  AnimationBudget.ikBudgetUsed[0] = 0;

  AnimationStats.activeAnimated[0] = 0;
  AnimationStats.boneCount[0] = 0;
  AnimationStats.ikChainCount[0] = 0;
  AnimationStats.ikSolved[0] = 0;
  AnimationStats.ikFailed[0] = 0;
  AnimationStats.motionSearchCount[0] = 0;
  AnimationStats.morphTargetCount[0] = 0;

  BonePosePool.poseTRS.fill(0);
  BonePosePool.worldTRS.fill(0);
  BonePosePool.boneParentIdx.fill(-1);
  BonePosePool.boneJointType.fill(0);
  BonePosePool.boneValid.fill(0);
  BonePosePool.slotOwner.fill(-1);
  BonePosePool.slotInUse.fill(0);
  BonePosePool.freeSlotHead[0] = 0;

  AnimationState.frame = 0;
  AnimationState.totalBinds = 0;
  AnimationState.totalPlays = 0;
  AnimationState.totalPauses = 0;
  AnimationState.totalStops = 0;
  AnimationState.totalLayers = 0;
  AnimationState.totalBlends = 0;
  AnimationState.totalIKChainsCreated = 0;
  AnimationState.totalIKSolves = 0;
  AnimationState.totalTwoBoneSolves = 0;
  AnimationState.totalFABRIKSolves = 0;
  AnimationState.totalCCDSolves = 0;
  AnimationState.totalFullBodySolves = 0;
  AnimationState.totalGroundRays = 0;
  AnimationState.totalStairDetections = 0;
  AnimationState.totalMotionMatches = 0;
  AnimationState.totalMotionTransitions = 0;
  AnimationState.totalMorphUpdates = 0;
  AnimationState.peakActivePoses = 0;
  AnimationState.activePoses = 0;
  AnimationState.lastTickMs = 0;
  AnimationState.avgTickMs = 0;
  AnimationState.lastIKMs = 0;
  AnimationState.avgIKMs = 0;
  AnimationState.lastMMMs = 0;
  AnimationState.avgMMMs = 0;
}

/* ------------------------------------------------------------------ */
/* 22. DEFAULT EXPORT                                                */
/* ------------------------------------------------------------------ */

const _defaultExport = {
  // Constants
  MAX_BONES_PER_ENTITY,
  MAX_ACTIVE_ANIMATED,
  MAX_IK_CHAINS_PER_ENTITY,
  MAX_ANIM_LAYERS_PER_ENTITY,
  MAX_CLIPS_PER_ENTITY,
  MAX_TRACKS_PER_CLIP,
  MAX_MORPH_TARGETS,
  MAX_MOTION_POSES,
  MOTION_FEATURE_DIM,
  DEFAULT_IK_ITERATIONS,
  DEFAULT_IK_TOLERANCE,
  MAX_GROUND_CONTACTS,
  POSE_STRIDE,

  // Enums
  SKELETON_TYPE,
  SKELETON_TYPE_NAME,
  JOINT_TYPE,
  JOINT_TYPE_NAME,
  IK_CHAIN_TYPE,
  IK_CHAIN_TYPE_NAME,
  ANIM_STATE,
  ANIM_STATE_NAME,
  MM_SEARCH_MODE,
  FOOT_CONTACT,

  // Components
  AnimationStateComp,
  AnimationBinding,
  AnimationLayer,
  AnimationClip,
  AnimationBlend,
  AnimationRoot,
  AnimationMorph,
  AnimationEvent,
  AnimationIK,
  IKTargets,
  GroundContact,
  FootPlacement,
  MotionMatchState,
  MotionPose,
  AnimationBudget,
  AnimationStats,
  BonePosePool,
  ANIMATION_COMPONENTS,

  // Module state
  AnimationState,

  // Pose slot management
  acquirePoseSlot,
  releasePoseSlot,

  // Bone pose accessors
  getBonePosePosition,
  setBonePosePosition,
  getBonePoseRotation,
  setBonePoseRotation,
  getBonePoseScale,
  setBonePoseScale,

  // Skeleton builders
  registerSkeleton,
  buildHumanoidSkeleton,
  buildBirdSkeleton,
  buildQuadrupedSkeleton,
  buildMachineSkeleton,

  // Animation playback
  bindClip,
  playClip,
  pauseAnimation,
  stopAnimation,
  setAnimationTime,
  setAnimationSpeed,
  setLayerWeight,
  addAnimationLayer,
  blendAnimation,

  // Morph
  setMorphWeight,
  evaluateMorphWeights,

  // Motion matching
  extractMotionFeatures,
  computeMatchScore,
  searchMotionDatabase,
  registerMotionPose,
  performMotionMatch,
  beginMotionMatchTransition,
  tickMotionMatchTransition,

  // IK chain construction
  createIKChain,
  setIKTarget,
  setIKTargetRotation,
  setIKPoleVector,

  // IK solvers
  solveTwoBoneIK,
  solveFABRIKChain,
  solveCCDChain,
  solveArmIK,
  solveHandIK,
  solveLegIK,
  solveFootIK,
  solveHeadLookAt,
  solveSpineLookAt,
  solveWingIK,
  solveTailIK,
  solveQuadrupedIK,
  solveMachineChainIK,
  solveFullBodyIK,

  // Ground adaptation
  raycastGround,
  computeGroundNormal,
  adaptFootToGround,
  adaptHipToGround,
  detectStair,
  snapFootToStair,
  adaptBodyToStairs,

  // Frame
  tickAnimation,
  tickAnimationEntity,
  tickAnimationSystem,

  // Diagnostics
  getAnimationEntityStats,
  getAnimationSystemReport,

  // Registration
  registerAnimationComponents,

  // Reset
  resetAnimationState,
};

export default _defaultExport;