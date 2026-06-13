// audio_stream_player_3d.cpp
#include "audio_stream_player_3d.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <limits>

namespace lighting {

// ============================================================================
// SPCAP panning algorithm (based on Godot original)
// ============================================================================
struct Spcap {
    struct Speaker {
        float x, y, z;           // normalized direction
        float eff_speakers;      // precomputed effective number of speakers
        mutable float sq_gain;    // temporary storage for gain
    };
    std::vector<Speaker> speakers;

    Spcap(int count, const float* dirs) {
        speakers.resize(count);
        for (int i = 0; i < count; ++i) {
            speakers[i].x = dirs[i*3];
            speakers[i].y = dirs[i*3+1];
            speakers[i].z = dirs[i*3+2];
            speakers[i].eff_speakers = 0.0f;
            speakers[i].sq_gain = 0.0f;
        }
        // precompute effective number of speakers
        for (size_t i = 0; i < speakers.size(); ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < speakers.size(); ++j) {
                float dot = speakers[i].x * speakers[j].x +
                            speakers[i].y * speakers[j].y +
                            speakers[i].z * speakers[j].z;
                sum += 0.5f * (1.0f + dot);
            }
            speakers[i].eff_speakers = sum;
        }
    }

    void compute_gains(const float* source_dir, float* out_gains) const {
        // SPCAP algorithm: G_i = ( (S·D_i + 1)/2 )^a / E_i
        // with a = 2, E_i = effective number of speakers
        for (size_t i = 0; i < speakers.size(); ++i) {
            float dot = source_dir[0] * speakers[i].x +
                        source_dir[1] * speakers[i].y +
                        source_dir[2] * speakers[i].z;
            float weight = (dot + 1.0f) * 0.5f;
            weight = weight * weight; // squared
            out_gains[i] = weight / (speakers[i].eff_speakers + 1e-6f);
        }
        // normalize
        float sum = 0.0f;
        for (size_t i = 0; i < speakers.size(); ++i) sum += out_gains[i];
        if (sum > 1e-6f) {
            for (size_t i = 0; i < speakers.size(); ++i) out_gains[i] /= sum;
        }
    }
};

// ============================================================================
// VelocityTracker3D for doppler (simplified)
// ============================================================================
class VelocityTracker3D {
public:
    void update(const double* pos) {
        if (first) {
            memcpy(prev_pos, pos, 3*sizeof(double));
            first = false;
            return;
        }
        // linear velocity = delta position / delta time
        // we need delta time from caller; store separately
        double dx = pos[0] - prev_pos[0];
        double dy = pos[1] - prev_pos[1];
        double dz = pos[2] - prev_pos[2];
        // we will compute velocity each frame; for simplicity we just update prev
        memcpy(prev_pos, pos, 3*sizeof(double));
        // store displacement (to be used with delta)
        vel_disp[0] = dx; vel_disp[1] = dy; vel_disp[2] = dz;
    }
    void get_linear_velocity(double* out, double delta) const {
        if (delta < 1e-6) {
            out[0]=out[1]=out[2]=0.0;
            return;
        }
        out[0] = vel_disp[0] / delta;
        out[1] = vel_disp[1] / delta;
        out[2] = vel_disp[2] / delta;
    }
private:
    bool first = true;
    double prev_pos[3] = {0,0,0};
    mutable double vel_disp[3] = {0,0,0};
};

// ============================================================================
// Internal audio playback (simplified – would use actual audio server)
// ============================================================================
class AudioStreamPlayback {
public:
    virtual void start(float from_pos) = 0;
    virtual void stop() = 0;
    virtual bool is_playing() const = 0;
    virtual void set_paused(bool paused) = 0;
    virtual void mix(float* buffer, int frames, int channels, float volume) = 0;
    virtual ~AudioStreamPlayback() = default;
};

// ============================================================================
// AudioStream stub (to be replaced with actual resource)
// ============================================================================
class AudioStream {
public:
    virtual AudioStreamPlayback* instantiate() const = 0;
    virtual ~AudioStream() = default;
};

// ============================================================================
// AudioStreamPlayer3D implementation
// ============================================================================
struct AudioStreamPlayer3D::Impl {
    AudioStream* stream = nullptr;
    AudioStreamPlayback* playback = nullptr;

    float volume_db = 0.0f;          // dB
    float pitch_scale = 1.0f;
    float max_db = 3.0f;             // maximum volume (dB) before clamping
    AttenuationModel att_model = AttenuationModel::INVERSE_DISTANCE;
    float max_distance = 1000.0f;
    float unit_size = 10.0f;
    int out_of_range_mode = 0;       // 0=mix, 1=pause, 2=disable

    PanningMethod pan_method = PanningMethod::STEREO;
    float panning_strength = 1.0f;
    std::vector<float> speaker_positions; // 3 per speaker

    DopplerTracking doppler_tracking = DopplerTracking::DISABLED;
    float doppler_strength = 1.0f;
    VelocityTracker3D velocity_tracker;

    char bus_name[64] = "Master";
    uint32_t area_mask = 0xFFFFFFFF;
    bool reverb_zone_enabled = true;

    // runtime panning gains (for up to 8 output channels)
    float pan_gains[8] = {1.0f,1.0f,0,0,0,0,0,0};

    // cached world listener transform (set by audio server)
    double listener_pos[3] = {0,0,0};
    double listener_vel[3] = {0,0,0};
    double listener_forward[3] = {0,0,-1};

    // SPCAP object if needed
    std::unique_ptr<Spcap> spcap;

    // physics area intersection (for reverb)
    std::vector<Area3D*> overlapping_areas;
    float last_reverb_vol = 0.0f;

    // internal mix state
    uint64_t last_mix_count = 0;
    bool force_update_panning = true;
    bool playing = false;

    void update_panning(const double* source_dir);
    float compute_attenuation(float distance) const;
    float compute_doppler_pitch() const;
};

AudioStreamPlayer3D::AudioStreamPlayer3D() : pimpl(std::make_unique<Impl>()) {}
AudioStreamPlayer3D::~AudioStreamPlayer3D() {
    if (pimpl->playback) delete pimpl->playback;
}

void AudioStreamPlayer3D::set_stream(AudioStream* stream) {
    if (pimpl->playback) {
        delete pimpl->playback;
        pimpl->playback = nullptr;
    }
    pimpl->stream = stream;
    if (stream) {
        pimpl->playback = stream->instantiate();
    }
}
AudioStream* AudioStreamPlayer3D::get_stream() const { return pimpl->stream; }

void AudioStreamPlayer3D::play(float from_pos) {
    if (pimpl->playback) {
        pimpl->playback->start(from_pos);
        pimpl->playing = true;
    }
}
void AudioStreamPlayer3D::stop() {
    if (pimpl->playback) {
        pimpl->playback->stop();
        pimpl->playing = false;
    }
}
bool AudioStreamPlayer3D::is_playing() const { return pimpl->playing; }
void AudioStreamPlayer3D::set_paused(bool paused) {
    if (pimpl->playback) pimpl->playback->set_paused(paused);
}
bool AudioStreamPlayer3D::is_paused() const { return false; }

void AudioStreamPlayer3D::set_volume_db(float db) { pimpl->volume_db = db; }
float AudioStreamPlayer3D::get_volume_db() const { return pimpl->volume_db; }
void AudioStreamPlayer3D::set_pitch_scale(float scale) { pimpl->pitch_scale = scale; }
float AudioStreamPlayer3D::get_pitch_scale() const { return pimpl->pitch_scale; }
void AudioStreamPlayer3D::set_max_db(float max_db) { pimpl->max_db = max_db; }
float AudioStreamPlayer3D::get_max_db() const { return pimpl->max_db; }

void AudioStreamPlayer3D::set_attenuation_model(AttenuationModel model) { pimpl->att_model = model; }
AttenuationModel AudioStreamPlayer3D::get_attenuation_model() const { return pimpl->att_model; }
void AudioStreamPlayer3D::set_max_distance(float distance) { pimpl->max_distance = distance; }
float AudioStreamPlayer3D::get_max_distance() const { return pimpl->max_distance; }
void AudioStreamPlayer3D::set_unit_size(float size) { pimpl->unit_size = size; }
float AudioStreamPlayer3D::get_unit_size() const { return pimpl->unit_size; }
void AudioStreamPlayer3D::set_out_of_range_mode(int mode) { pimpl->out_of_range_mode = mode; }
int AudioStreamPlayer3D::get_out_of_range_mode() const { return pimpl->out_of_range_mode; }

void AudioStreamPlayer3D::set_panning_method(PanningMethod method) {
    pimpl->pan_method = method;
    if (method == PanningMethod::SPCAP && !pimpl->spcap) {
        // default 7.1 speaker layout
        float default_dirs[24] = {
            1,0,0, -1,0,0, 0,1,0, 0,-1,0, 0,0,1, 0,0,-1,
            0.707f,0.707f,0, -0.707f,0.707f,0, 0.707f,-0.707f,0, -0.707f,-0.707f,0
        };
        pimpl->spcap = std::make_unique<Spcap>(8, default_dirs);
    }
    pimpl->force_update_panning = true;
}
PanningMethod AudioStreamPlayer3D::get_panning_method() const { return pimpl->pan_method; }

void AudioStreamPlayer3D::set_panning_strength(float strength) { pimpl->panning_strength = strength; }
float AudioStreamPlayer3D::get_panning_strength() const { return pimpl->panning_strength; }

void AudioStreamPlayer3D::set_custom_speaker_positions(const std::vector<float>& directions) {
    if (directions.size() % 3 == 0 && directions.size() >= 6) {
        pimpl->speaker_positions = directions;
        if (pimpl->pan_method == PanningMethod::SPCAP) {
            pimpl->spcap = std::make_unique<Spcap>((int)directions.size()/3, directions.data());
            pimpl->force_update_panning = true;
        }
    }
}

void AudioStreamPlayer3D::set_doppler_tracking(DopplerTracking tracking) { pimpl->doppler_tracking = tracking; }
DopplerTracking AudioStreamPlayer3D::get_doppler_tracking() const { return pimpl->doppler_tracking; }
void AudioStreamPlayer3D::set_doppler_strength(float strength) { pimpl->doppler_strength = strength; }
float AudioStreamPlayer3D::get_doppler_strength() const { return pimpl->doppler_strength; }

void AudioStreamPlayer3D::set_bus(const char* bus_name) {
    strncpy(pimpl->bus_name, bus_name, 63);
    pimpl->bus_name[63] = 0;
}
const char* AudioStreamPlayer3D::get_bus() const { return pimpl->bus_name; }
void AudioStreamPlayer3D::set_area_mask(uint32_t mask) { pimpl->area_mask = mask; }
uint32_t AudioStreamPlayer3D::get_area_mask() const { return pimpl->area_mask; }
void AudioStreamPlayer3D::set_reverb_zone_enabled(bool enabled) { pimpl->reverb_zone_enabled = enabled; }
bool AudioStreamPlayer3D::is_reverb_zone_enabled() const { return pimpl->reverb_zone_enabled; }

void AudioStreamPlayer3D::Impl::update_panning(const double* source_dir) {
    if (!force_update_panning) return;
    force_update_panning = false;

    // Convert source direction to unit vector
    double dx = source_dir[0];
    double dy = source_dir[1];
    double dz = source_dir[2];
    double len = sqrt(dx*dx + dy*dy + dz*dz);
    if (len > 1e-6) {
        dx /= len; dy /= len; dz /= len;
    } else {
        dx = 0; dy = 0; dz = 1;
    }

    if (pan_method == PanningMethod::STEREO) {
        // simple stereo panning using sin/cos
        // assume listener facing -Z, source direction relative to listener forward
        // compute angle in XZ plane
        double angle = atan2(dx, -dz);
        float left = cos(angle);
        float right = sin(angle);
        // apply strength
        left = left * panning_strength + (1.0f - panning_strength);
        right = right * panning_strength + (1.0f - panning_strength);
        pan_gains[0] = left;
        pan_gains[1] = right;
    } else if (pan_method == PanningMethod::SPCAP && spcap) {
        float dir[3] = {(float)dx, (float)dy, (float)dz};
        spcap->compute_gains(dir, pan_gains);
    } else {
        // default: equal gain for first two channels
        pan_gains[0] = 1.0f;
        pan_gains[1] = 1.0f;
    }
}

float AudioStreamPlayer3D::Impl::compute_attenuation(float distance) const {
    switch (att_model) {
        case AttenuationModel::INVERSE_DISTANCE:
            distance = std::max(distance, unit_size);
            return unit_size / distance;
        case AttenuationModel::INVERSE_SQUARE_DISTANCE:
            distance = std::max(distance, unit_size);
            return (unit_size * unit_size) / (distance * distance);
        case AttenuationModel::LOGARITHMIC:
            if (distance <= unit_size) return 1.0f;
            return unit_size / distance; // not truly logarithmic, but approximation
        case AttenuationModel::DISABLED:
        default:
            return 1.0f;
    }
}

float AudioStreamPlayer3D::Impl::compute_doppler_pitch() const {
    if (doppler_tracking == DopplerTracking::DISABLED) return 1.0f;
    // speed of sound = 340.0 m/s (default)
    const float sound_speed = 340.0f;
    // relative velocity along listener->source direction
    double dx = listener_pos[0] - get_global_transform().origin[0];
    double dy = listener_pos[1] - get_global_transform().origin[1];
    double dz = listener_pos[2] - get_global_transform().origin[2];
    double dist = sqrt(dx*dx + dy*dy + dz*dz);
    if (dist < 1e-6) return 1.0f;
    double dir[3] = {dx/dist, dy/dist, dz/dist};
    double source_vel[3];
    // we don't have source velocity directly, use from physics?
    // simplify: assume listener velocity only
    double v_listener = listener_vel[0]*dir[0] + listener_vel[1]*dir[1] + listener_vel[2]*dir[2];
    double v_source = 0.0; // approximate stationary
    double relative_vel = v_listener - v_source;
    float pitch_factor = sound_speed / (sound_speed + relative_vel);
    return pitch_factor * doppler_strength + (1.0f - doppler_strength);
}

void AudioStreamPlayer3D::sync_to_audio_server(double delta_time) {
    if (!pimpl->playback || !pimpl->playing) return;

    // get source position (world)
    Transform3D global = get_global_transform();
    double source_pos[3] = {global.origin[0], global.origin[1], global.origin[2]};

    // update doppler velocity tracker if needed
    if (pimpl->doppler_tracking == DopplerTracking::IDLE_STEP) {
        pimpl->velocity_tracker.update(source_pos);
    }

    // compute distance to listener
    double dx = source_pos[0] - pimpl->listener_pos[0];
    double dy = source_pos[1] - pimpl->listener_pos[1];
    double dz = source_pos[2] - pimpl->listener_pos[2];
    float dist = (float)sqrt(dx*dx + dy*dy + dz*dz);

    // apply out-of-range behavior
    if (dist > pimpl->max_distance && pimpl->out_of_range_mode != 0) {
        if (pimpl->out_of_range_mode == 1) {
            pimpl->playback->set_paused(true);
        } else if (pimpl->out_of_range_mode == 2) {
            pimpl->playing = false;
            pimpl->playback->stop();
        }
        return;
    }

    // compute attenuation
    float att = pimpl->compute_attenuation(dist);
    if (att > 1.0f) att = 1.0f;
    float linear_vol = pow(10.0f, pimpl->volume_db / 20.0f);
    linear_vol *= att;

    // clamp to max_db
    float max_linear = pow(10.0f, pimpl->max_db / 20.0f);
    if (linear_vol > max_linear) linear_vol = max_linear;

    // compute source direction relative to listener
    double source_dir[3] = {dx / (dist+1e-6), dy / (dist+1e-6), dz / (dist+1e-6)};
    pimpl->update_panning(source_dir);

    // compute doppler pitch
    float pitch = pimpl->pitch_scale;
    if (pimpl->doppler_tracking != DopplerTracking::DISABLED) {
        double source_vel[3];
        if (pimpl->doppler_tracking == DopplerTracking::IDLE_STEP) {
            pimpl->velocity_tracker.get_linear_velocity(source_vel, delta_time);
        } else {
            // for physics step we would fetch from body
            source_vel[0]=source_vel[1]=source_vel[2]=0.0;
        }
        // compute relative velocity
        double rel_vel = (pimpl->listener_vel[0]-source_vel[0])*source_dir[0] +
                         (pimpl->listener_vel[1]-source_vel[1])*source_dir[1] +
                         (pimpl->listener_vel[2]-source_vel[2])*source_dir[2];
        const float sound_speed = 340.0f;
        pitch *= (sound_speed) / (sound_speed + (float)rel_vel);
    }
    if (pitch < 0.01f) pitch = 0.01f;

    // store gains for mixing (audio server will call mix_audio later)
    // For now, we just store them in pimpl for the next mix call
}

void AudioStreamPlayer3D::mix_audio(float* buffer, int frames, int channels) {
    if (!pimpl->playback || !pimpl->playing) {
        if (buffer) memset(buffer, 0, frames * channels * sizeof(float));
        return;
    }

    // temporary mix buffer (mono or stereo from playback)
    const int max_frames = 1024;
    float mix_buf[max_frames * 2] = {0};
    int out_channels = std::min(channels, 8); // limit to 8 output channels
    int in_channels = 2; // assume stereo source

    for (int frame = 0; frame < frames; frame += max_frames) {
        int block = std::min(max_frames, frames - frame);
        // get audio from playback
        pimpl->playback->mix(mix_buf, block, in_channels, 1.0f);

        // apply panning gains and volume
        for (int i = 0; i < block; ++i) {
            for (int ch = 0; ch < out_channels; ++ch) {
                float gain = (ch < 8) ? pimpl->pan_gains[ch] : 0.0f;
                float sample = 0.0f;
                if (in_channels == 1) {
                    sample = mix_buf[i] * gain;
                } else {
                    // assume stereo: left = mix_buf[i*2], right = mix_buf[i*2+1]
                    sample = (ch == 0) ? mix_buf[i*2] * gain : mix_buf[i*2+1] * gain;
                }
                buffer[(frame+i)*out_channels + ch] += sample;
            }
        }
    }
}

void AudioStreamPlayer3D::ready() {
    Node3D::ready();
}

void AudioStreamPlayer3D::process(double delta) {
    Node3D::process(delta);
    // sync with audio server
    sync_to_audio_server(delta);
}

void AudioStreamPlayer3D::synchronize_render_server(double delta) {
    Node3D::synchronize_render_server(delta);
    // nothing extra needed
}

} // namespace lighting