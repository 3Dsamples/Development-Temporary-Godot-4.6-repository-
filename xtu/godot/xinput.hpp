// godot/xinput.hpp

#ifndef XTENSOR_XINPUT_HPP
#define XTENSOR_XINPUT_HPP

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xexpression.hpp"
#include "../containers/xarray.hpp"
#include "../containers/xtensor.hpp"
#include "../core/xview.hpp"
#include "../core/xfunction.hpp"
#include "../math/xstats.hpp"
#include "../math/xnorm.hpp"
#include "../math/xrandom.hpp"
#include "xvariant.hpp"
#include "xclassdb.hpp"
#include "xnode.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <map>
#include <unordered_map>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <algorithm>
#include <numeric>
#include <functional>
#include <cmath>
#include <queue>
#include <mutex>

#if XTENSOR_GODOT_CLASSDB_AVAILABLE
    #include <godot_cpp/classes/input.hpp>
    #include <godot_cpp/classes/input_map.hpp>
    #include <godot_cpp/classes/input_event.hpp>
    #include <godot_cpp/classes/input_event_key.hpp>
    #include <godot_cpp/classes/input_event_mouse_button.hpp>
    #include <godot_cpp/classes/input_event_mouse_motion.hpp>
    #include <godot_cpp/classes/input_event_joypad_button.hpp>
    #include <godot_cpp/classes/input_event_joypad_motion.hpp>
    #include <godot_cpp/classes/input_event_screen_touch.hpp>
    #include <godot_cpp/classes/input_event_screen_drag.hpp>
    #include <godot_cpp/classes/input_event_action.hpp>
    #include <godot_cpp/classes/input_event_gesture.hpp>
    #include <godot_cpp/classes/input_event_magnify_gesture.hpp>
    #include <godot_cpp/classes/input_event_pan_gesture.hpp>
    #include <godot_cpp/classes/input_event_midi.hpp>
    #include <godot_cpp/classes/engine.hpp>
    #include <godot_cpp/classes/os.hpp>
    #include <godot_cpp/classes/display_server.hpp>
    #include <godot_cpp/variant/utility_functions.hpp>
    #include <godot_cpp/variant/vector2.hpp>
    #include <godot_cpp/variant/vector3.hpp>
#endif

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        namespace godot_bridge
        {
            // --------------------------------------------------------------------
            // Input State Tensors (batch query for multiple entities)
            // --------------------------------------------------------------------
            using InputStateTensor = xarray_container<float>; // N x M (entities x input_dim)

            enum class InputType
            {
                KEYBOARD = 0,
                MOUSE_BUTTON = 1,
                MOUSE_AXIS = 2,
                JOYPAD_BUTTON = 3,
                JOYPAD_AXIS = 4,
                ACTION = 5,
                TOUCH = 6,
                GESTURE = 7,
                MIDI = 8,
                SENSOR = 9
            };

            struct InputBinding
            {
                InputType type;
                int device_id = 0;      // 0 for all, -1 for any
                int button_index = 0;   // key code, button index, or action hash
                int axis_index = 0;
                float scale = 1.0f;
                float deadzone = 0.2f;
                bool inverted = false;
                std::string action_name;
            };

            // --------------------------------------------------------------------
            // XInputManager - Tensor-based input processing
            // --------------------------------------------------------------------
            class XInputManager
            {
            public:
                using InputBuffer = std::vector<godot::Ref<godot::InputEvent>>;

                XInputManager()
                {
                    // Initialize device counts
                    m_keyboard_count = 1;
                    m_mouse_count = 1;
                    m_joypad_count = 8;
                    m_touch_count = 10;
                }

                // Collect all input events for the current frame
                void collect_events()
                {
#if XTENSOR_GODOT_CLASSDB_AVAILABLE
                    godot::Input* input = godot::Input::get_singleton();
                    if (!input) return;
                    
                    m_current_events.clear();
                    // Note: Godot doesn't expose a "get_all_events" API.
                    // Events must be accumulated via _input() callbacks.
                    // For batch processing, we rely on users pushing events manually
                    // or using the accumulated buffer.
#endif
                }

                void push_event(const godot::Ref<godot::InputEvent>& event)
                {
                    m_current_events.push_back(event);
                }

                void clear_events()
                {
                    m_current_events.clear();
                }

                // Batch query: map N entities to their bound input values
                // bindings: N x max_bindings_per_entity
                // Returns: N x (max_bindings_per_entity) tensor of input values
                xarray_container<float> query_bindings(const std::vector<std::vector<InputBinding>>& bindings) const
                {
                    size_t n_entities = bindings.size();
                    size_t max_bindings = 0;
                    for (const auto& b : bindings)
                        max_bindings = std::max(max_bindings, b.size());
                    
                    xarray_container<float> result({n_entities, max_bindings}, 0.0f);
                    
                    for (size_t i = 0; i < n_entities; ++i)
                    {
                        for (size_t j = 0; j < bindings[i].size(); ++j)
                        {
                            result(i, j) = get_binding_value(bindings[i][j]);
                        }
                    }
                    return result;
                }

                // Batch query: map N entities to a set of named actions
                // action_names: vector of action strings to query
                // Returns: N x M tensor of action states (0.0 = released, 1.0 = pressed, or axis value)
                xarray_container<float> query_actions(size_t n_entities, const std::vector<std::string>& action_names) const
                {
                    size_t m = action_names.size();
                    xarray_container<float> result({n_entities, m}, 0.0f);
                    
                    // For now, actions are global (not per-entity). 
                    // Per-entity action mapping would require device ID assignment.
                    for (size_t j = 0; j < m; ++j)
                    {
                        float val = get_action_value(action_names[j]);
                        for (size_t i = 0; i < n_entities; ++i)
                            result(i, j) = val;
                    }
                    return result;
                }

                // Batch query: get joystick states for all connected joypads
                // Returns: J x A (joypads x axes) tensor
                xarray_container<float> query_all_joypads() const
                {
#if XTENSOR_GODOT_CLASSDB_AVAILABLE
                    godot::Input* input = godot::Input::get_singleton();
                    if (!input) return {};
                    
                    auto joypads = input->get_connected_joypads();
                    size_t n_joypads = static_cast<size_t>(joypads.size());
                    const size_t n_axes = 8; // Standard: left X/Y, right X/Y, L2/R2, etc.
                    const size_t n_buttons = 16;
                    
                    xarray_container<float> axes({n_joypads, n_axes}, 0.0f);
                    xarray_container<float> buttons({n_joypads, n_buttons}, 0.0f);
                    
                    for (size_t i = 0; i < n_joypads; ++i)
                    {
                        int id = joypads[i];
                        for (size_t a = 0; a < n_axes; ++a)
                        {
                            axes(i, a) = input->get_joy_axis(id, static_cast<godot::JoyAxis>(a));
                        }
                        for (size_t b = 0; b < n_buttons; ++b)
                        {
                            buttons(i, b) = input->is_joy_button_pressed(id, static_cast<godot::JoyButton>(b)) ? 1.0f : 0.0f;
                        }
                    }
                    // Return concatenated or separate? We'll just return axes for simplicity
                    return axes;
#else
                    return {};
#endif
                }

                // Batch query: get all keyboard key states as a tensor
                // Returns: K-length vector (1=pressed, 0=released)
                xarray_container<float> query_keyboard_state() const
                {
#if XTENSOR_GODOT_CLASSDB_AVAILABLE
                    godot::Input* input = godot::Input::get_singleton();
                    if (!input) return {};
                    
                    const size_t num_keys = 256; // Approximate
                    xarray_container<float> result({num_keys}, 0.0f);
                    for (size_t k = 0; k < num_keys; ++k)
                    {
                        result(k) = input->is_key_pressed(static_cast<godot::Key>(k)) ? 1.0f : 0.0f;
                    }
                    return result;
#else
                    return {};
#endif
                }

                // Mouse state tensor: [pos_x, pos_y, delta_x, delta_y, wheel_x, wheel_y, left, right, middle, ...]
                xarray_container<float> query_mouse_state() const
                {
#if XTENSOR_GODOT_CLASSDB_AVAILABLE
                    godot::Input* input = godot::Input::get_singleton();
                    if (!input) return {};
                    
                    xarray_container<float> result({12}, 0.0f);
                    godot::Vector2 pos = input->get_mouse_position();
                    godot::Vector2 rel = input->get_last_mouse_velocity();
                    result(0) = pos.x;
                    result(1) = pos.y;
                    result(2) = rel.x;
                    result(3) = rel.y;
                    result(4) = 0.0f; // wheel not easily accessible in batch
                    result(5) = 0.0f;
                    result(6) = input->is_mouse_button_pressed(godot::MOUSE_BUTTON_LEFT) ? 1.0f : 0.0f;
                    result(7) = input->is_mouse_button_pressed(godot::MOUSE_BUTTON_RIGHT) ? 1.0f : 0.0f;
                    result(8) = input->is_mouse_button_pressed(godot::MOUSE_BUTTON_MIDDLE) ? 1.0f : 0.0f;
                    result(9) = input->is_mouse_button_pressed(godot::MOUSE_BUTTON_WHEEL_UP) ? 1.0f : 0.0f;
                    result(10) = input->is_mouse_button_pressed(godot::MOUSE_BUTTON_WHEEL_DOWN) ? 1.0f : 0.0f;
                    result(11) = input->is_mouse_button_pressed(godot::MOUSE_BUTTON_WHEEL_LEFT) ? 1.0f : 0.0f;
                    return result;
#else
                    return {};
#endif
                }

                // Touch state tensor: T x 4 (pos_x, pos_y, pressure, phase)
                xarray_container<float> query_touch_state() const
                {
#if XTENSOR_GODOT_CLASSDB_AVAILABLE
                    godot::Input* input = godot::Input::get_singleton();
                    if (!input) return {};
                    
                    // Godot doesn't provide a direct API to get all touches.
                    // We accumulate via event buffer.
                    std::map<int, godot::Vector2> active_touches;
                    std::map<int, float> touch_pressures;
                    for (const auto& ev : m_current_events)
                    {
                        if (ev->is_class("InputEventScreenTouch"))
                        {
                            godot::Ref<godot::InputEventScreenTouch> touch = ev;
                            int idx = touch->get_index();
                            if (touch->is_pressed())
                            {
                                active_touches[idx] = touch->get_position();
                                touch_pressures[idx] = touch->get_pressure();
                            }
                            else
                            {
                                active_touches.erase(idx);
                                touch_pressures.erase(idx);
                            }
                        }
                        else if (ev->is_class("InputEventScreenDrag"))
                        {
                            godot::Ref<godot::InputEventScreenDrag> drag = ev;
                            int idx = drag->get_index();
                            active_touches[idx] = drag->get_position();
                            touch_pressures[idx] = drag->get_pressure();
                        }
                    }
                    
                    size_t n_touches = active_touches.size();
                    xarray_container<float> result({n_touches, 4}, 0.0f);
                    size_t i = 0;
                    for (const auto& p : active_touches)
                    {
                        result(i, 0) = p.second.x;
                        result(i, 1) = p.second.y;
                        result(i, 2) = touch_pressures[p.first];
                        result(i, 3) = 1.0f; // phase: 0=began,1=moved,2=stationary,3=ended,4=cancelled (simplified)
                        ++i;
                    }
                    return result;
#else
                    return {};
#endif
                }

                // Gesture state: G x 4 (type, pos_x, pos_y, value)
                xarray_container<float> query_gesture_state() const
                {
                    // Accumulated from events
                    return {};
                }

                // Record input sequence for replay or analysis
                struct InputFrame
                {
                    uint64_t timestamp_usec;
                    xarray_container<float> keyboard;
                    xarray_container<float> mouse;
                    xarray_container<float> joypads_axes;
                    xarray_container<float> joypads_buttons;
                    xarray_container<float> touches;
                };
                
                std::vector<InputFrame> record_frames(size_t num_frames, float fps = 60.0f)
                {
                    std::vector<InputFrame> frames;
                    // Would require time-based sampling - not fully implemented here.
                    return frames;
                }

                // Replay recorded input frames (simulate input)
                void replay_frame(const InputFrame& frame)
                {
                    // Set internal state or generate synthetic events.
                }

                // --------------------------------------------------------------------
                // Binding Management
                // --------------------------------------------------------------------
                void add_binding(const std::string& name, const InputBinding& binding)
                {
                    m_bindings[name].push_back(binding);
                }

                void set_bindings(const std::string& name, const std::vector<InputBinding>& bindings)
                {
                    m_bindings[name] = bindings;
                }

                std::vector<InputBinding> get_bindings(const std::string& name) const
                {
                    auto it = m_bindings.find(name);
                    if (it != m_bindings.end())
                        return it->second;
                    return {};
                }

                float get_binding_value(const std::string& name) const
                {
                    auto it = m_bindings.find(name);
                    if (it == m_bindings.end()) return 0.0f;
                    float max_val = 0.0f;
                    for (const auto& b : it->second)
                    {
                        float v = get_binding_value(b);
                        if (std::abs(v) > std::abs(max_val))
                            max_val = v;
                    }
                    return max_val;
                }

                float get_binding_value(const InputBinding& binding) const
                {
#if XTENSOR_GODOT_CLASSDB_AVAILABLE
                    godot::Input* input = godot::Input::get_singleton();
                    if (!input) return 0.0f;
                    
                    float raw = 0.0f;
                    switch (binding.type)
                    {
                        case InputType::KEYBOARD:
                            raw = input->is_key_pressed(static_cast<godot::Key>(binding.button_index)) ? 1.0f : 0.0f;
                            break;
                        case InputType::MOUSE_BUTTON:
                            raw = input->is_mouse_button_pressed(static_cast<godot::MouseButton>(binding.button_index)) ? 1.0f : 0.0f;
                            break;
                        case InputType::MOUSE_AXIS:
                            {
                                godot::Vector2 pos = input->get_mouse_position();
                                godot::Vector2 rel = input->get_last_mouse_velocity();
                                if (binding.axis_index == 0) raw = pos.x;
                                else if (binding.axis_index == 1) raw = pos.y;
                                else if (binding.axis_index == 2) raw = rel.x;
                                else if (binding.axis_index == 3) raw = rel.y;
                            }
                            break;
                        case InputType::JOYPAD_BUTTON:
                            raw = input->is_joy_button_pressed(binding.device_id, static_cast<godot::JoyButton>(binding.button_index)) ? 1.0f : 0.0f;
                            break;
                        case InputType::JOYPAD_AXIS:
                            raw = input->get_joy_axis(binding.device_id, static_cast<godot::JoyAxis>(binding.axis_index));
                            break;
                        case InputType::ACTION:
                            raw = input->get_action_strength(godot::StringName(binding.action_name.c_str()));
                            break;
                        case InputType::TOUCH:
                            // Touch requires event accumulation
                            break;
                        default:
                            break;
                    }
                    
                    // Apply deadzone
                    if (std::abs(raw) < binding.deadzone)
                        raw = 0.0f;
                    else
                        raw = (raw > 0 ? raw - binding.deadzone : raw + binding.deadzone) / (1.0f - binding.deadzone);
                    
                    if (binding.inverted)
                        raw = -raw;
                    
                    return raw * binding.scale;
#else
                    return 0.0f;
#endif
                }

                float get_action_value(const std::string& action_name) const
                {
#if XTENSOR_GODOT_CLASSDB_AVAILABLE
                    godot::Input* input = godot::Input::get_singleton();
                    if (!input) return 0.0f;
                    return input->get_action_strength(godot::StringName(action_name.c_str()));
#else
                    return 0.0f;
#endif
                }

                // --------------------------------------------------------------------
                // Input Mapping Tensor (for AI agents)
                // --------------------------------------------------------------------
                // Create an observation tensor for reinforcement learning agents
                // Format: [keyboard(256), mouse(12), joypads_axes(8*8), joypads_buttons(16*8), touches(10*4)]
                xarray_container<float> create_observation_tensor() const
                {
                    std::vector<float> obs;
                    
                    auto keyboard = query_keyboard_state();
                    for (size_t i = 0; i < keyboard.size(); ++i)
                        obs.push_back(keyboard(i));
                    
                    auto mouse = query_mouse_state();
                    for (size_t i = 0; i < mouse.size(); ++i)
                        obs.push_back(mouse(i));
                    
                    auto joypads = query_all_joypads();
                    for (size_t i = 0; i < joypads.size(); ++i)
                        obs.push_back(joypads.flat(i));
                    
                    auto touches = query_touch_state();
                    for (size_t i = 0; i < touches.size(); ++i)
                        obs.push_back(touches.flat(i));
                    
                    xarray_container<float> result({obs.size()});
                    std::copy(obs.begin(), obs.end(), result.begin());
                    return result;
                }

            private:
                InputBuffer m_current_events;
                std::map<std::string, std::vector<InputBinding>> m_bindings;
                size_t m_keyboard_count = 1;
                size_t m_mouse_count = 1;
                size_t m_joypad_count = 8;
                size_t m_touch_count = 10;
            };

            // --------------------------------------------------------------------
            // XInputNode - Godot node for tensor-based input
            // --------------------------------------------------------------------
#if XTENSOR_GODOT_CLASSDB_AVAILABLE
            class XInputNode : public godot::Node
            {
                GDCLASS(XInputNode, godot::Node)

            private:
                XInputManager m_manager;
                godot::Ref<XTensorNode> m_observation_tensor;
                godot::Ref<XTensorNode> m_action_bindings_tensor;
                godot::Dictionary m_binding_configs;
                bool m_auto_capture = true;
                bool m_record_input = false;
                std::vector<XInputManager::InputFrame> m_recorded_frames;
                size_t m_record_max_frames = 3600; // 60 seconds at 60fps

            protected:
                static void _bind_methods()
                {
                    godot::ClassDB::bind_method(godot::D_METHOD("set_observation_tensor", "tensor"), &XInputNode::set_observation_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_observation_tensor"), &XInputNode::get_observation_tensor);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_auto_capture", "enabled"), &XInputNode::set_auto_capture);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_auto_capture"), &XInputNode::get_auto_capture);
                    godot::ClassDB::bind_method(godot::D_METHOD("set_record_input", "enabled"), &XInputNode::set_record_input);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_record_input"), &XInputNode::get_record_input);
                    
                    godot::ClassDB::bind_method(godot::D_METHOD("add_binding", "name", "config"), &XInputNode::add_binding);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_binding_value", "name"), &XInputNode::get_binding_value);
                    godot::ClassDB::bind_method(godot::D_METHOD("query_observation"), &XInputNode::query_observation);
                    godot::ClassDB::bind_method(godot::D_METHOD("query_action_batch", "actions"), &XInputNode::query_action_batch);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_connected_joypads"), &XInputNode::get_connected_joypads);
                    godot::ClassDB::bind_method(godot::D_METHOD("is_action_pressed", "action"), &XInputNode::is_action_pressed);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_action_strength", "action"), &XInputNode::get_action_strength);
                    
                    godot::ClassDB::bind_method(godot::D_METHOD("start_recording"), &XInputNode::start_recording);
                    godot::ClassDB::bind_method(godot::D_METHOD("stop_recording"), &XInputNode::stop_recording);
                    godot::ClassDB::bind_method(godot::D_METHOD("get_recorded_frames"), &XInputNode::get_recorded_frames);
                    godot::ClassDB::bind_method(godot::D_METHOD("save_recording", "path"), &XInputNode::save_recording);
                    godot::ClassDB::bind_method(godot::D_METHOD("load_recording", "path"), &XInputNode::load_recording);
                    
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::OBJECT, "observation_tensor", godot::PROPERTY_HINT_RESOURCE_TYPE, "XTensorResource"), "set_observation_tensor", "get_observation_tensor");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::BOOL, "auto_capture"), "set_auto_capture", "get_auto_capture");
                    ADD_PROPERTY(godot::PropertyInfo(godot::Variant::BOOL, "record_input"), "set_record_input", "get_record_input");
                    
                    ADD_SIGNAL(godot::MethodInfo("input_captured"));
                    ADD_SIGNAL(godot::MethodInfo("recording_started"));
                    ADD_SIGNAL(godot::MethodInfo("recording_stopped"));
                }

            public:
                XInputNode() {}
                
                void _ready() override
                {
                    set_process_input(true);
                    set_process(true);
                    if (!m_observation_tensor.is_valid())
                        m_observation_tensor.instantiate();
                }
                
                void _input(const godot::Ref<godot::InputEvent>& event) override
                {
                    if (m_auto_capture)
                        m_manager.push_event(event);
                }
                
                void _process(double delta) override
                {
                    if (m_auto_capture)
                    {
                        query_observation();
                        m_manager.clear_events();
                    }
                    
                    if (m_record_input)
                    {
                        record_frame();
                    }
                }

                void set_observation_tensor(const godot::Ref<XTensorNode>& tensor) { m_observation_tensor = tensor; }
                godot::Ref<XTensorNode> get_observation_tensor() const { return m_observation_tensor; }
                void set_auto_capture(bool enabled) { m_auto_capture = enabled; }
                bool get_auto_capture() const { return m_auto_capture; }
                void set_record_input(bool enabled) { m_record_input = enabled; }
                bool get_record_input() const { return m_record_input; }

                void add_binding(const godot::String& name, const godot::Dictionary& config)
                {
                    InputBinding b;
                    if (config.has("type"))
                        b.type = static_cast<InputType>(static_cast<int>(config["type"]));
                    if (config.has("device_id"))
                        b.device_id = config["device_id"];
                    if (config.has("button_index"))
                        b.button_index = config["button_index"];
                    if (config.has("axis_index"))
                        b.axis_index = config["axis_index"];
                    if (config.has("scale"))
                        b.scale = config["scale"];
                    if (config.has("deadzone"))
                        b.deadzone = config["deadzone"];
                    if (config.has("inverted"))
                        b.inverted = config["inverted"];
                    if (config.has("action_name"))
                        b.action_name = config["action_name"].operator godot::String().utf8().get_data();
                    
                    m_manager.add_binding(name.utf8().get_data(), b);
                    m_binding_configs[name] = config;
                }

                float get_binding_value(const godot::String& name) const
                {
                    return m_manager.get_binding_value(name.utf8().get_data());
                }

                void query_observation()
                {
                    auto obs = m_manager.create_observation_tensor();
                    if (m_observation_tensor.is_valid())
                    {
                        m_observation_tensor->set_data(XVariant::from_xarray(obs.cast<double>()).variant());
                        emit_signal("input_captured");
                    }
                }

                godot::Ref<XTensorNode> query_action_batch(const godot::PackedStringArray& actions) const
                {
                    auto result = XTensorNode::create_zeros(godot::PackedInt64Array());
                    std::vector<std::string> action_vec;
                    for (int i = 0; i < actions.size(); ++i)
                        action_vec.push_back(actions[i].utf8().get_data());
                    
                    auto batch = m_manager.query_actions(1, action_vec);
                    result->set_data(XVariant::from_xarray(batch.cast<double>()).variant());
                    return result;
                }

                godot::PackedInt32Array get_connected_joypads() const
                {
#if XTENSOR_GODOT_CLASSDB_AVAILABLE
                    godot::Input* input = godot::Input::get_singleton();
                    if (input)
                        return input->get_connected_joypads();
#endif
                    return godot::PackedInt32Array();
                }

                bool is_action_pressed(const godot::String& action) const
                {
#if XTENSOR_GODOT_CLASSDB_AVAILABLE
                    godot::Input* input = godot::Input::get_singleton();
                    if (input)
                        return input->is_action_pressed(godot::StringName(action));
#endif
                    return false;
                }

                float get_action_strength(const godot::String& action) const
                {
#if XTENSOR_GODOT_CLASSDB_AVAILABLE
                    godot::Input* input = godot::Input::get_singleton();
                    if (input)
                        return input->get_action_strength(godot::StringName(action));
#endif
                    return 0.0f;
                }

                void start_recording()
                {
                    m_record_input = true;
                    m_recorded_frames.clear();
                    emit_signal("recording_started");
                }

                void stop_recording()
                {
                    m_record_input = false;
                    emit_signal("recording_stopped");
                }

                godot::Array get_recorded_frames() const
                {
                    godot::Array arr;
                    for (const auto& frame : m_recorded_frames)
                    {
                        godot::Dictionary dict;
                        dict["timestamp"] = frame.timestamp_usec;
                        godot::Ref<XTensorNode> kbd;
                        kbd.instantiate();
                        kbd->set_data(XVariant::from_xarray(frame.keyboard.cast<double>()).variant());
                        dict["keyboard"] = kbd;
                        // ... other fields
                        arr.append(dict);
                    }
                    return arr;
                }

                void save_recording(const godot::String& path)
                {
                    // Serialize recorded frames to JSON or binary
                    JsonArchive ar;
                    // ... convert frames to JSON
                    ar.save(path.utf8().get_data());
                }

                void load_recording(const godot::String& path)
                {
                    JsonArchive ar;
                    ar.load(path.utf8().get_data());
                    // ... parse back to frames
                }

            private:
                void record_frame()
                {
                    XInputManager::InputFrame frame;
                    frame.timestamp_usec = static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::steady_clock::now().time_since_epoch()).count());
                    frame.keyboard = m_manager.query_keyboard_state();
                    frame.mouse = m_manager.query_mouse_state();
                    frame.joypads_axes = m_manager.query_all_joypads();
                    // ... other states
                    
                    m_recorded_frames.push_back(frame);
                    if (m_recorded_frames.size() > m_record_max_frames)
                        m_recorded_frames.erase(m_recorded_frames.begin());
                }
            };
#endif // XTENSOR_GODOT_CLASSDB_AVAILABLE

            // --------------------------------------------------------------------
            // Registration
            // --------------------------------------------------------------------
            class XInputRegistry
            {
            public:
                static void register_classes()
                {
#if XTENSOR_GODOT_CLASSDB_AVAILABLE
                    godot::ClassDB::register_class<XInputNode>();
#endif
                }
            };

        } // namespace godot_bridge

        using godot_bridge::InputType;
        using godot_bridge::InputBinding;
        using godot_bridge::XInputManager;
        using godot_bridge::XInputNode;
        using godot_bridge::XInputRegistry;

    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XINPUT_HPP

// godot/xinput.hpp