// File : 8
// name : src/shaders/008_atmosphereShaders.js
// description : High-performance GLSL vertex and fragment shaders for the atmospheric glow sphere. Optimized for Android mobile GPUs with mediump precision, efficient Fresnel calculations, and directional light asymmetry. Excludes nebula and lens flare effects to maintain strict real-time performance budgets.
// License : Glpt3
// A new version using the https://github.com/PubDeer/astro-loop/tree/main

import {
  ShaderMaterial,
  Vector3,
  Color,
  FrontSide,
  AdditiveBlending
} from 'three';

export const atmosphereVertexShader = `
  precision mediump float;
  
  uniform mat4 modelViewMatrix;
  uniform mat4 projectionMatrix;
  uniform mat3 normalMatrix;
  
  attribute vec3 position;
  attribute vec3 normal;
  
  varying vec3 vWorldNormal;
  varying vec3 vViewPosition;
  
  void main() {
    vWorldNormal = normalize(normalMatrix * normal);
    
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    vViewPosition = -mvPosition.xyz;
    
    gl_Position = projectionMatrix * mvPosition;
  }
`;

export const atmosphereFragmentShader = `
  precision mediump float;
  
  uniform vec3 uGlowColor;
  uniform float uGlowIntensity;
  uniform float uFresnelPower;
  uniform float uAsymmetryFactor;
  uniform vec3 uLightDir;
  
  varying vec3 vWorldNormal;
  varying vec3 vViewPosition;
  
  void main() {
    vec3 viewDir = normalize(vViewPosition);
    vec3 worldNormal = normalize(vWorldNormal);
    vec3 lightDir = normalize(uLightDir);
    
    float viewDotNormal = max(dot(viewDir, worldNormal), 0.0);
    float fresnel = 1.0 - viewDotNormal;
    
    float fresnelPow = pow(fresnel, uFresnelPower);
    
    float lightInfluence = max(dot(worldNormal, lightDir), 0.0);
    float finalInfluence = mix(uAsymmetryFactor, 1.0, lightInfluence);
    
    float alpha = fresnelPow * uGlowIntensity * finalInfluence;
    
    if (alpha < 0.015) {
      discard;
    }
    
    gl_FragColor = vec4(uGlowColor * finalInfluence, alpha);
  }
`;

const _tempColor = new Color();
const _tempVec3 = new Vector3();

export function createAtmosphereMaterial(uniformsData) {
  return new ShaderMaterial({
    vertexShader: atmosphereVertexShader,
    fragmentShader: atmosphereFragmentShader,
    uniforms: {
      uGlowColor: { value: new Color(uniformsData.glowColorR, uniformsData.glowColorG, uniformsData.glowColorB) },
      uGlowIntensity: { value: uniformsData.glowIntensity },
      uFresnelPower: { value: uniformsData.fresnelPower },
      uAsymmetryFactor: { value: uniformsData.asymmetryFactor },
      uLightDir: { value: _tempVec3.set(uniformsData.lightDirX, uniformsData.lightDirY, uniformsData.lightDirZ).normalize() }
    },
    side: FrontSide,
    blending: AdditiveBlending,
    transparent: true,
    depthWrite: false,
    depthTest: true
  });
}

export function updateAtmosphereMaterialUniforms(material, uniformsData) {
  if (!material || !material.uniforms) return;
  
  const u = material.uniforms;
  
  u.uGlowColor.value.setRGB(uniformsData.glowColorR, uniformsData.glowColorG, uniformsData.glowColorB);
  u.uGlowIntensity.value = uniformsData.glowIntensity;
  u.uFresnelPower.value = uniformsData.fresnelPower;
  u.uAsymmetryFactor.value = uniformsData.asymmetryFactor;
  u.uLightDir.value.set(uniformsData.lightDirX, uniformsData.lightDirY, uniformsData.lightDirZ).normalize();
}

export default {
  atmosphereVertexShader,
  atmosphereFragmentShader,
  createAtmosphereMaterial,
  updateAtmosphereMaterialUniforms
};