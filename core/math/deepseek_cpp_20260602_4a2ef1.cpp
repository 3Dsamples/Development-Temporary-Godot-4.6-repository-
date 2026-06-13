//File 0112 : core/parallel/parallel_pipeline.h
//Data‑parallel pipeline: stages applied in order to a token buffer, each stage executed in parallel using parallel_for (bulk synchronous), with optional serial per‑stage override.
#ifndef CORE_PARALLEL_PARALLEL_PIPELINE_H
#define CORE_PARALLEL_PARALLEL_PIPELINE_H

#include "parallel_for.h"
#include <vector>
#include <functional>
#include <type_traits>

namespace SimulationMath {
namespace parallel {

// -----------------------------------------------------------------------------
// 1. Pipeline class – processes a vector of tokens through a chain of stages.
//    Each stage is a callable that accepts a reference to a token.
//    By default stages are applied in parallel across all tokens (using parallel_for).
//    Set `parallel = false` to force sequential execution for that stage.
// -----------------------------------------------------------------------------
template <typename Token>
class Pipeline {
public:
    using StageFunc = std::function<void(Token&)>;

    // Add a stage. If `parallel` is true (default), the stage is applied
    // to all tokens concurrently using parallel_for.
    Pipeline& add_stage(StageFunc func, bool parallel = true) noexcept {
        stages_.push_back({std::move(func), parallel});
        return *this;
    }

    // Process the vector of tokens through all stages in order.
    void run(std::vector<Token>& tokens) noexcept {
        for (const auto& stage : stages_) {
            if (stage.parallel) {
                // parallel over all tokens
                parallel_for(tokens.size(), [&](size_t i) {
                    stage.func(tokens[i]);
                });
            } else {
                // sequential (useful if the stage has global side‑effects that must be ordered)
                for (size_t i = 0; i < tokens.size(); ++i) {
                    stage.func(tokens[i]);
                }
            }
        }
    }

private:
    struct Stage {
        StageFunc func;
        bool parallel;
    };
    std::vector<Stage> stages_;
};

// -----------------------------------------------------------------------------
// 2. Pipeline that also supports a final output consumer – the last stage may
//    push results to an output container without modifying the original tokens.
//    This is a convenience wrapper.
// -----------------------------------------------------------------------------
template <typename Token, typename Output>
class PipelineWithOutput {
public:
    using StageFunc = std::function<void(const Token&, Output&)>;

    PipelineWithOutput& add_stage(StageFunc func, bool parallel = true) noexcept {
        stages_.push_back({std::move(func), parallel});
        return *this;
    }

    // Process each token, collecting outputs into the provided vector.
    // The size of `outputs` must match `tokens.size()`.
    void run(const std::vector<Token>& tokens, std::vector<Output>& outputs) noexcept {
        if (outputs.size() != tokens.size()) outputs.resize(tokens.size());
        for (const auto& stage : stages_) {
            if (stage.parallel) {
                parallel_for(tokens.size(), [&](size_t i) {
                    stage.func(tokens[i], outputs[i]);
                });
            } else {
                for (size_t i = 0; i < tokens.size(); ++i) {
                    stage.func(tokens[i], outputs[i]);
                }
            }
        }
    }

private:
    struct Stage {
        StageFunc func;
        bool parallel;
    };
    std::vector<Stage> stages_;
};

} // namespace parallel
} // namespace SimulationMath

#endif // CORE_PARALLEL_PARALLEL_PIPELINE_H