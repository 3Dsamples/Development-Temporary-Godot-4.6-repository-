// File : 12
// name : src/shaders/012_starFieldShaders.js
// description : High-performance GLSL vertex and fragment shaders for the procedural star field particle system. Strictly imports and injects normal quantization logic from mesh/normalQuantizer.js to ensure architectural consistency and GPU-side normal tweaking for stylized star volume. Implements GPU-driven twinkling, distance-based size attenuation, and early alpha discard for maximum Android mobile fill-rate efficiency. Excludes nebula and lens flare.
// License : Glpt3
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import {
  ShaderMaterial,
  Vector3,
  AdditiveBlending
} from 'three';
import { getAnimeNormalQuantizationGLSL } from '../mesh/006_normalQuantizer.js';

const starFieldVertexShaderCore = `
  precision mediump float;
  
  uniform float uTime;
  uniform float uPixelRatio;
  uniform vec3 uCameraPosition;
  
  attribute float size;
  attribute float phase;
  attribute vec3 normal;
  
  varying float vAlpha;
  varying float vBrightness;
  varying vec3 vViewNormal;
  
  void main() {
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    vec3 transformedNormal = normalize(normalMatrix * normal);
    
    // Inject shared anime normal quantization for stylized star volume consistency
    ${getAnimeNormalQuantizationGLSL('transformedNormal', 'vec3(0.0, 1.0, 0.0)', '4.0')}
    
    vViewNormal = transformedNormal;
    
    float dist = length(mvPosition.xyz);
    float attenuation = 150.0 / max(dist, 1.0);
    
    gl_PointSize = clamp(size * uPixelRatio * attenuation, 1.0, 64.0);
    gl_Position = projectionMatrix * mvPosition;
    
    float twinkle = 0.6 + 0.4 * sin(uTime * 1.5 + phase * 6.28318);
    vAlpha = twinkle;
    vBrightness = twinkle;
  }
`;

const starFieldFragmentShaderCore = `
  precision mediump float;
  
  varying float vAlpha;
  varying float vBrightness;
  varying vec3 vViewNormal;
  
  void main() {
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    
    if (dist > 0.5) {
      discard;
    }
    
    float glow = 1.0 - (dist * 2.0);
    glow = pow(glow, 1.8);
    
    vec3 viewDir = vec3(0.0, 0.0, 1.0);
    float normalDot = max(dot(vViewNormal, viewDir), 0.0);
    
    float quantizedNormal = floor(normalDot * 4.0) / 4.0;
    float normalModulation = mix(0.8, 1.0, quantizedNormal);
    
    vec3 starColor = vec3(0.85, 0.92, 1.0) * vBrightness * normalModulation;
    
    float finalAlpha = glow * vAlpha;
    
    if (finalAlpha < 0.05) {
      discard;
    }
    
    gl_FragColor = vec4(starColor, finalAlpha);
  }
`;

const _tempVec3 = new Vector3();

export function createStarFieldMaterial(uniformsData) {
  return new ShaderMaterial({
    vertexShader: starFieldVertexShaderCore,
    fragmentShader: starFieldFragmentShaderCore,
    uniforms: {
      uTime: { value: 0.0 },
      uPixelRatio: { value: uniformsData.pixelRatio || 1.0 },
      uCameraPosition: { value: _tempVec3.copy(uniformsData.cameraPosition || new Vector3(0, 0, 0)) }
    },
    transparent: true,
    blending: AdditiveBlending,
    depthWrite: false,
    depthTest: true
  });
}

export function updateStarFieldMaterialUniforms(material, time, pixelRatio, cameraPosition) {
  if (!material || !material.uniforms) return;
  
  const u = material.uniforms;
  
  u.uTime.value = time;
  u.uPixelRatio.value = pixelRatio;
  u.uCameraPosition.value.copy(cameraPosition);
}

export function createFallbackStarMaterial(size, pixelRatio) {
  return new ShaderMaterial({
    vertexShader: `
      precision mediump float;
      uniform float uPixelRatio;
      attribute float size;
      void main() {
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = size * uPixelRatio * (150.0 / max(length(mvPosition.xyz), 1.0));
        gl_Position = projectionMatrix * mvPosition;
      }
    `,
    fragmentShader: `
      precision mediump float;
      void main() {
        vec2 coord = gl_PointCoord - vec2(0.5);
        if (length(coord) > 0.5) discard;
        gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);
      }
    `,
    uniforms: {
      uPixelRatio: { value: pixelRatio }
    },
    transparent: true,
    blending: AdditiveBlending,
    depthWrite: false,
    depthTest: true
  });
}

export function disposeStarFieldMaterial(material) {
  if (material) {
    if (material.uniforms) {
      if (material.uniforms.uCameraPosition) {
        material.uniforms.uCameraPosition.value = null;
      }
    }
    material.dispose();
  }
}

export default {
  starFieldVertexShaderCore,
  starFieldFragmentShaderCore,
  createStarFieldMaterial,
  updateStarFieldMaterialUniforms,
  createFallbackStarMaterial,
  disposeStarFieldMaterial
};