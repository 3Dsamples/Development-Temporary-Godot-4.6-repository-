// File : 10
// name : src/shaders/010_animePlanetShaders.js
// description : High-performance GLSL vertex and fragment shaders for the procedural anime planet. Strictly imports and injects GLSL noise strings from utils/noise.js and normal quantization logic from mesh/normalQuantizer.js to maintain DRY principles and perfect CPU/GPU logic alignment. Optimized for Android mobile GPUs with mediump precision, minimal dynamic branching, and efficient math operations. Excludes nebula and lens flare.
// License : Glpt3
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import {
  ShaderMaterial,
  Vector3,
  Color,
  FrontSide
} from 'three';
import { glslNoise } from '../../utils/005_noise.js';
import { 
  getAnimeNormalQuantizationGLSL, 
  getAnimeRimNormalExaggerationGLSL 
} from '../../mesh/006_normalQuantizer.js';

const vertexShaderCore = `
  precision mediump float;
  
  uniform mat4 modelMatrix;
  uniform mat4 modelViewMatrix;
  uniform mat4 projectionMatrix;
  uniform mat3 normalMatrix;
  uniform vec3 uLightDir;
  uniform float uShadingSteps;
  
  attribute vec3 position;
  attribute vec3 normal;
  
  varying vec3 vWorldPosition;
  varying vec3 vWorldNormal;
  varying vec3 vViewPosition;
  varying vec3 vTriplanarCoords;
  varying float vLightDot;
  
  void main() {
    vec4 worldPos = modelMatrix * vec4(position, 1.0);
    vWorldPosition = worldPos.xyz;
    
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    vViewPosition = -mvPosition.xyz;
    
    vTriplanarCoords = position;
    
    vec3 transformedNormal = normalize(normalMatrix * normal);
    
    ${getAnimeNormalQuantizationGLSL('transformedNormal', 'uLightDir', 'uShadingSteps')}
    
    vWorldNormal = transformedNormal;
    vLightDot = dot(vWorldNormal, normalize(uLightDir));
    
    gl_Position = projectionMatrix * mvPosition;
  }
`;

const fragmentShaderCore = `
  precision mediump float;
  
  uniform float uTime;
  uniform vec3 uColorTop;
  uniform vec3 uColorBottom;
  uniform float uSwirlScale;
  uniform float uSwirlSpeed;
  uniform float uGoldThreshold;
  uniform float uCyanWispThreshold;
  uniform float uRimPower;
  uniform float uSpecularStep;
  uniform vec3 uLightDir;
  uniform vec3 uRimColor;
  uniform float uEnableRimLight;
  uniform float uRimThreshold;
  
  varying vec3 vWorldPosition;
  varying vec3 vWorldNormal;
  varying vec3 vViewPosition;
  varying vec3 vTriplanarCoords;
  varying float vLightDot;
  
  ${glslNoise}
  
  void main() {
    vec3 p = vTriplanarCoords * uSwirlScale;
    p.y += uTime * uSwirlSpeed;
    
    float n1 = fbm(p, 4.0);
    
    vec3 warp = vec3(
      fbm(p + n1, 4.0), 
      fbm(p + n1 + 100.0, 4.0), 
      fbm(p + n1 + 200.0, 4.0)
    );
    
    float n2 = fbm(p + warp * 1.5, 4.0);
    
    float lat = (vWorldPosition.y + 1.0) * 0.5;
    vec3 baseColor = mix(uColorBottom, uColorTop, smoothstep(0.2, 0.8, lat));
    
    vec3 finalColor = baseColor;
    float swirlMask = smoothstep(0.4, 0.6, n2);
    finalColor = mix(finalColor, baseColor * 1.3, swirlMask * 0.5);
    
    float goldMask = smoothstep(uGoldThreshold, uGoldThreshold + 0.1, n2);
    float goldLight = max(vLightDot, 0.0) * 1.5;
    finalColor = mix(finalColor, vec3(1.0, 0.8, 0.3), goldMask * goldLight);
    
    float cyanMask = smoothstep(uCyanWispThreshold, uCyanWispThreshold + 0.05, n2);
    finalColor = mix(finalColor, vec3(0.4, 0.9, 1.0), cyanMask * 0.8);
    
    vec3 viewDir = normalize(vViewPosition);
    float rim = 1.0 - max(dot(viewDir, vWorldNormal), 0.0);
    rim = smoothstep(uRimThreshold, uRimThreshold + 0.2, rim);
    rim = pow(rim, uRimPower);
    
    float lightRim = max(dot(vWorldNormal, normalize(uLightDir)), 0.0);
    finalColor += uRimColor * rim * lightRim * 1.5 * uEnableRimLight;
    
    vec3 reflectDir = reflect(-normalize(uLightDir), vWorldNormal);
    float spec = pow(max(dot(reflectDir, viewDir), 0.0), 32.0);
    spec = step(uSpecularStep, spec);
    finalColor += vec3(1.0, 0.95, 0.9) * spec * max(vLightDot, 0.0);
    
    float finalLight = (vLightDot * 0.5) + 0.5;
    finalColor *= finalLight;
    
    finalColor += baseColor * 0.15 * (1.0 - finalLight);
    
    gl_FragColor = vec4(finalColor, 1.0);
  }
`;

const _tempColorTop = new Color();
const _tempColorBottom = new Color();
const _tempRimColor = new Color();
const _tempLightDir = new Vector3();

export function createAnimePlanetMaterial(shaderData, quantizeData) {
  return new ShaderMaterial({
    vertexShader: vertexShaderCore,
    fragmentShader: fragmentShaderCore,
    uniforms: {
      uTime: { value: 0.0 },
      uLightDir: { value: _tempLightDir.set(shaderData.lightDirX, shaderData.lightDirY, shaderData.lightDirZ).normalize() },
      uColorTop: { value: _tempColorTop.setRGB(shaderData.colorTopR, shaderData.colorTopG, shaderData.colorTopB) },
      uColorBottom: { value: _tempColorBottom.setRGB(shaderData.colorBottomR, shaderData.colorBottomG, shaderData.colorBottomB) },
      uSwirlScale: { value: shaderData.swirlScale },
      uSwirlSpeed: { value: shaderData.swirlSpeed },
      uGoldThreshold: { value: shaderData.goldThreshold },
      uCyanWispThreshold: { value: shaderData.cyanWispThreshold },
      uRimPower: { value: shaderData.rimPower },
      uSpecularStep: { value: shaderData.specularStep },
      uShadingSteps: { value: quantizeData.shadingSteps },
      uRimThreshold: { value: quantizeData.rimThreshold },
      uEnableRimLight: { value: quantizeData.enableRimLight },
      uRimColor: { value: _tempRimColor.setRGB(quantizeData.rimColorR, quantizeData.rimColorG, quantizeData.rimColorB) }
    },
    side: FrontSide
  });
}

export function updateAnimePlanetMaterialUniforms(material, shaderData, quantizeData) {
  if (!material || !material.uniforms) return;
  
  const u = material.uniforms;
  
  u.uTime.value = shaderData.time;
  u.uLightDir.value.set(shaderData.lightDirX, shaderData.lightDirY, shaderData.lightDirZ).normalize();
  u.uColorTop.value.setRGB(shaderData.colorTopR, shaderData.colorTopG, shaderData.colorTopB);
  u.uColorBottom.value.setRGB(shaderData.colorBottomR, shaderData.colorBottomG, shaderData.colorBottomB);
  u.uSwirlScale.value = shaderData.swirlScale;
  u.uSwirlSpeed.value = shaderData.swirlSpeed;
  u.uGoldThreshold.value = shaderData.goldThreshold;
  u.uCyanWispThreshold.value = shaderData.cyanWispThreshold;
  u.uRimPower.value = shaderData.rimPower;
  u.uSpecularStep.value = shaderData.specularStep;
  
  u.uShadingSteps.value = quantizeData.shadingSteps;
  u.uRimThreshold.value = quantizeData.rimThreshold;
  u.uEnableRimLight.value = quantizeData.enableRimLight;
  u.uRimColor.value.setRGB(quantizeData.rimColorR, quantizeData.rimColorG, quantizeData.rimColorB);
}

export default {
  vertexShaderCore,
  fragmentShaderCore,
  createAnimePlanetMaterial,
  updateAnimePlanetMaterialUniforms
};