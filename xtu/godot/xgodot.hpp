// include/xtu/godot/xgodot.hpp
// xtensor-unified - Master include for Godot 4.6 integration
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_GODOT_XGODOT_HPP
#define XTU_GODOT_XGODOT_HPP

// #############################################################################
// Core Godot types and utilities
// #############################################################################
#include "xtu/godot/xcore.hpp"           // String, FileAccess, OS, Thread, containers
#include "xtu/godot/xvariant.hpp"        // Variant, Dictionary, Array, packed arrays
#include "xtu/godot/xclassdb.hpp"        // ClassDB, Object, RefCounted, MethodBind

// #############################################################################
// Resource and scene system
// #############################################################################
#include "xtu/godot/xresource.hpp"       // Resource, ResourceLoader, ResourceSaver
#include "xtu/godot/xnode.hpp"           // Node, SceneTree, Viewport, Node2D, Node3D

// #############################################################################
// Physics
// #############################################################################
#include "xtu/godot/xphysics2d.hpp"      // PhysicsServer2D, RigidBody2D, Area2D
#include "xtu/godot/xphysics3d.hpp"      // PhysicsServer3D, RigidBody3D, CharacterBody3D
#include "xtu/godot/xsoftbody3d.hpp"     // SoftBody3D

// #############################################################################
// Rendering and graphics
// #############################################################################
#include "xtu/godot/xrenderingserver.hpp" // RenderingServer, RenderingDevice
#include "xtu/godot/xlighting.hpp"        // Light3D, LightmapGI, VoxelGI, ReflectionProbe
#include "xtu/godot/xshader.hpp"          // Shader, ShaderMaterial, VisualShader
#include "xtu/godot/xocclusion.hpp"       // Occluder3D, Room, Portal, RoomManager
#include "xtu/godot/xdecals.hpp"          // Decal, DecalTexture
#include "xtu/godot/xsky.hpp"             // Sky, Environment, WorldEnvironment

// #############################################################################
// Audio
// #############################################################################
#include "xtu/godot/xaudioserver.hpp"    // AudioServer, AudioStream, AudioStreamPlayer
#include "xtu/godot/xaudioeffects.hpp"   // AudioEffect, AudioEffectReverb, EQ, etc.

// #############################################################################
// Input and GUI
// #############################################################################
#include "xtu/godot/xinput.hpp"          // Input, InputMap, InputEvent
#include "xtu/godot/xgui.hpp"            // Control, Button, Label, Tree, etc.

// #############################################################################
// Animation and IK
// #############################################################################
#include "xtu/godot/xanimation.hpp"      // Animation, AnimationPlayer, AnimationTree
#include "xtu/godot/xik3d.hpp"           // SkeletonIK3D, FABRIK, CCD, TwoBoneIK

// #############################################################################
// Scripting
// #############################################################################
#include "xtu/godot/xgdscript.hpp"       // GDScript, GDScriptFunction, GDScriptVM

// #############################################################################
// Navigation and AI
// #############################################################################
#include "xtu/godot/xpathfinding.hpp"    // NavigationServer2D/3D, NavigationAgent

// #############################################################################
// 2D and 3D Nodes
// #############################################################################
#include "xtu/godot/xtilemap.hpp"        // TileMap, TileSet, TileMapLayer
#include "xtu/godot/xparticles2d.hpp"    // GPUParticles2D, ParticleProcessMaterial
#include "xtu/godot/xparticles3d.hpp"    // GPUParticles3D

// #############################################################################
// XR (AR/VR)
// #############################################################################
#include "xtu/godot/xxr.hpp"             // XRServer, XROrigin3D, XRController3D

// #############################################################################
// Multiplayer
// #############################################################################
#include "xtu/godot/xnetworking.hpp"     // MultiplayerAPI, ENetMultiplayerPeer, MultiplayerSpawner

// #############################################################################
// Editor (optional, conditionally included)
// #############################################################################
#ifdef XTU_GODOT_EDITOR_ENABLED
#include "xtu/godot/xeditor.hpp"          // EditorNode, EditorPlugin, EditorInspector
#include "xtu/godot/xeditor_import.hpp"   // ResourceImporter, EditorImportPlugin
#include "xtu/godot/xeditor_export.hpp"   // EditorExportPlatform, EditorExportPreset
#include "xtu/godot/xeditor_animation.hpp"// AnimationTrackEditor, AnimationBezierEditor
#include "xtu/godot/xeditor_script.hpp"   // CodeEditor, ScriptEditor, ScriptCreateDialog
#include "xtu/godot/xeditor_debugger.hpp" // ScriptEditorDebugger, EditorProfiler
#include "xtu/godot/xeditor_settings.hpp" // ProjectSettingsEditor, InputMapEditor
#include "xtu/godot/xeditor_docks.hpp"    // SceneTreeDock, InspectorDock, FileSystemDock
#include "xtu/godot/xeditor_pickers.hpp"  // EditorResourcePicker, EditorFileDialog
#include "xtu/godot/xeditor_theme.hpp"    // EditorTheme, EditorScale, EditorFonts
#include "xtu/godot/xeditor_history.hpp"  // EditorUndoRedoManager, VersionControlPlugin
#include "xtu/godot/xeditor_build.hpp"    // EditorBuildManager, EditorRun
#include "xtu/godot/xproject_manager.hpp" // ProjectManager
#endif

// #############################################################################
// Main application entry point (optional)
// #############################################################################
namespace xtu {
namespace godot {

class GodotApplication {
public:
    static int run(int argc, char* argv[]) {
        // Initialize core systems
        OS::get_singleton();
        Input::get_singleton();
        RenderingServer::get_singleton();
        PhysicsServer3D::get_singleton();
        AudioServer::get_singleton();

        // Parse command line
        bool editor_mode = false;
        String project_path;
        for (int i = 1; i < argc; ++i) {
            String arg(argv[i]);
            if (arg == "--editor") editor_mode = true;
            else if (arg == "--path" && i + 1 < argc) project_path = String(argv[++i]);
        }

        if (editor_mode) {
#ifdef XTU_GODOT_EDITOR_ENABLED
            // Launch editor
            if (!project_path.empty()) {
                // Open project
            }
            EditorNode* editor = new EditorNode();
            // Enter main loop
            while (true) {
                // Process events
                // Render
            }
#else
            std::cerr << "Editor support not compiled.\n";
            return 1;
#endif
        } else if (!project_path.empty()) {
            // Run project
            // Load main scene and start
        } else {
            // Launch Project Manager
#ifdef XTU_GODOT_EDITOR_ENABLED
            ProjectManager* pm = new ProjectManager();
            pm->show();
            // Main loop
#endif
        }

        return 0;
    }
};

} // namespace godot
} // namespace xtu

#endif // XTU_GODOT_XGODOT_HPP