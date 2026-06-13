// File 146: modules/genesis/src/grad/tape.h
// Reverse‑mode automatic differentiation tape.
// Records operations on Tensor values and performs backward gradient propagation.
// Supports basic arithmetic, activation functions, and vectorised operations.

#ifndef GENESIS_GRAD_TAPE_H
#define GENESIS_GRAD_TAPE_H

#include "tensor.h"
#include "core/templates/local_vector.h"
#include "core/typedefs.h"
#include <cmath>

namespace genesis::grad {

/**
 * Tape node representing a single primitive operation.
 */
struct TapeNode {
	enum Op : uint8_t {
		CONST = 0,
		ADD,
		SUB,
		MUL,
		DIV,
		RELU,
		SIGMOID,
		TANH,
		EXP,
		LOG,
		POW,      // y = x^c  (c is stored in extra)
		NEG
	};

	Op op;
	int left;           // index of left operand (or -1 if unary/const)
	int right;          // index of right operand (or -1 if unary/const)
	real_t value;       // cached output value
	real_t grad;        // adjoint accumulator
	real_t extra;       // storage for constant exponent, etc.
};

/**
 * The global gradient tape. At most one tape is active at a time (thread‑local
 * or scene‑local). Operations on Tensor automatically record on the active tape.
 */
class GradientTape {
private:
	LocalVector<TapeNode> nodes;
	bool recording;

public:
	GradientTape() : recording(false) {}

	// Start recording operations.
	void begin() { recording = true; nodes.clear(); }

	// Stop recording.
	void end() { recording = false; }

	// Ensure we are in recording mode.
	bool is_recording() const { return recording; }

	// Add a constant leaf node. Returns its index.
	int push_constant(real_t value) {
		TapeNode n = { TapeNode::CONST, -1, -1, value, 0.0, 0.0 };
		nodes.push_back(n);
		return nodes.size() - 1;
	}

	// Add a unary operation node. Returns its index.
	int push_unary(TapeNode::Op op, int input, real_t out_value, real_t extra = 0.0) {
		TapeNode n = { op, input, -1, out_value, 0.0, extra };
		nodes.push_back(n);
		return nodes.size() - 1;
	}

	// Add a binary operation node. Returns its index.
	int push_binary(TapeNode::Op op, int left, int right, real_t out_value) {
		TapeNode n = { op, left, right, out_value, 0.0, 0.0 };
		nodes.push_back(n);
		return nodes.size() - 1;
	}

	// Begin backward pass from a root node (typically a loss). Seeds its grad = 1.0.
	void backward(int root) {
		if (root < 0 || root >= nodes.size()) return;
		// Reset adjoints
		for (TapeNode &n : nodes) n.grad = 0.0;
		nodes[root].grad = 1.0;

		// Traverse in reverse topological order (nodes are appended in evaluation order).
		for (int i = nodes.size() - 1; i >= 0; --i) {
			const TapeNode &n = nodes[i];
			if (n.grad == 0.0) continue;
			switch (n.op) {
				case TapeNode::CONST: break; // no children
				case TapeNode::ADD:
					nodes[n.left].grad += n.grad;
					nodes[n.right].grad += n.grad;
					break;
				case TapeNode::SUB:
					nodes[n.left].grad += n.grad;
					nodes[n.right].grad -= n.grad;
					break;
				case TapeNode::MUL:
					nodes[n.left].grad += n.grad * nodes[n.right].value;
					nodes[n.right].grad += n.grad * nodes[n.left].value;
					break;
				case TapeNode::DIV: {
					real_t inv_right = 1.0 / nodes[n.right].value;
					real_t val_sq = nodes[n.right].value * nodes[n.right].value;
					nodes[n.left].grad += n.grad * inv_right;
					nodes[n.right].grad -= n.grad * nodes[n.left].value / val_sq;
				} break;
				case TapeNode::RELU:
					nodes[n.left].grad += n.grad * (nodes[n.left].value > 0.0 ? 1.0 : 0.0);
					break;
				case TapeNode::SIGMOID:
					real_t s = nodes[n.value].value;
					// s = 1/(1+exp(-x)), derivative = s*(1-s)
					nodes[n.left].grad += n.grad * s * (1.0 - s);
					break;
				case TapeNode::TANH: {
					real_t t = Math::tanh(nodes[n.left].value);
					nodes[n.left].grad += n.grad * (1.0 - t * t);
				} break;
				case TapeNode::EXP:
					nodes[n.left].grad += n.grad * nodes[n.value].value; // derivative = e^x = y
					break;
				case TapeNode::LOG:
					nodes[n.left].grad += n.grad / nodes[n.left].value; // derivative = 1/x
					break;
				case TapeNode::POW: // y = x^c, derivative = c * x^(c-1)
					real_t c = n.extra;
					nodes[n.left].grad += n.grad * c * Math::pow(nodes[n.left].value, c - 1.0);
					break;
				case TapeNode::NEG:
					nodes[n.left].grad -= n.grad;
					break;
			}
		}
	}

	// Retrieve the adjoint of a node (e.g., for parameter leaf).
	real_t get_gradient(int node_index) const {
		ERR_FAIL_INDEX_V(node_index, nodes.size(), 0.0);
		return nodes[node_index].grad;
	}
};

// Global tape instance (singleton per scene or thread).
// In a production engine, use thread‑local or tie to a World instance.
inline GradientTape *active_tape = nullptr;

// --- Overloaded Tensor operations that record on the active tape ---

inline int tape_id(const Tensor &t) { return t.tape_id; }

inline Tensor make_tensor(real_t value, int tape_idx) {
	Tensor t;
	t.value = value;
	t.grad = 0.0;
	t.tape_id = tape_idx;
	return t;
}

inline Tensor tensor_add(const Tensor &a, const Tensor &b) {
	if (!active_tape || !active_tape->is_recording()) return Tensor(a.value + b.value);
	int ia = a.tape_id, ib = b.tape_id;
	int out_idx = active_tape->push_binary(TapeNode::ADD, ia, ib, a.value + b.value);
	return make_tensor(a.value + b.value, out_idx);
}

inline Tensor tensor_sub(const Tensor &a, const Tensor &b) {
	if (!active_tape || !active_tape->is_recording()) return Tensor(a.value - b.value);
	int ia = a.tape_id, ib = b.tape_id;
	int out_idx = active_tape->push_binary(TapeNode::SUB, ia, ib, a.value - b.value);
	return make_tensor(a.value - b.value, out_idx);
}

inline Tensor tensor_mul(const Tensor &a, const Tensor &b) {
	if (!active_tape || !active_tape->is_recording()) return Tensor(a.value * b.value);
	int ia = a.tape_id, ib = b.tape_id;
	int out_idx = active_tape->push_binary(TapeNode::MUL, ia, ib, a.value * b.value);
	return make_tensor(a.value * b.value, out_idx);
}

inline Tensor tensor_div(const Tensor &a, const Tensor &b) {
	if (!active_tape || !active_tape->is_recording()) return Tensor(a.value / b.value);
	int ia = a.tape_id, ib = b.tape_id;
	int out_idx = active_tape->push_binary(TapeNode::DIV, ia, ib, a.value / b.value);
	return make_tensor(a.value / b.value, out_idx);
}

inline Tensor tensor_relu(const Tensor &x) {
	if (!active_tape || !active_tape->is_recording()) return Tensor(x.value > 0.0 ? x.value : 0.0);
	real_t out_val = x.value > 0.0 ? x.value : 0.0;
	int out_idx = active_tape->push_unary(TapeNode::RELU, x.tape_id, out_val);
	return make_tensor(out_val, out_idx);
}

inline Tensor tensor_sigmoid(const Tensor &x) {
	real_t s = 1.0 / (1.0 + Math::exp(-x.value));
	if (!active_tape || !active_tape->is_recording()) return Tensor(s);
	int out_idx = active_tape->push_unary(TapeNode::SIGMOID, x.tape_id, s);
	return make_tensor(s, out_idx);
}

inline Tensor tensor_tanh(const Tensor &x) {
	real_t t = Math::tanh(x.value);
	if (!active_tape || !active_tape->is_recording()) return Tensor(t);
	int out_idx = active_tape->push_unary(TapeNode::TANH, x.tape_id, t);
	return make_tensor(t, out_idx);
}

inline Tensor tensor_exp(const Tensor &x) {
	real_t e = Math::exp(x.value);
	if (!active_tape || !active_tape->is_recording()) return Tensor(e);
	int out_idx = active_tape->push_unary(TapeNode::EXP, x.tape_id, e);
	return make_tensor(e, out_idx);
}

inline Tensor tensor_log(const Tensor &x) {
	real_t l = Math::log(x.value);
	if (!active_tape || !active_tape->is_recording()) return Tensor(l);
	int out_idx = active_tape->push_unary(TapeNode::LOG, x.tape_id, l);
	return make_tensor(l, out_idx);
}

inline Tensor tensor_pow(const Tensor &x, real_t exponent) {
	real_t p = Math::pow(x.value, exponent);
	if (!active_tape || !active_tape->is_recording()) return Tensor(p);
	int out_idx = active_tape->push_unary(TapeNode::POW, x.tape_id, p, exponent);
	return make_tensor(p, out_idx);
}

inline Tensor tensor_neg(const Tensor &x) {
	if (!active_tape || !active_tape->is_recording()) return Tensor(-x.value);
	int out_idx = active_tape->push_unary(TapeNode::NEG, x.tape_id, -x.value);
	return make_tensor(-x.value, out_idx);
}

// Convenience global that returns a leaf constant recorded on the active tape.
inline Tensor constant(real_t value) {
	if (!active_tape || !active_tape->is_recording()) return Tensor(value);
	int idx = active_tape->push_constant(value);
	return make_tensor(value, idx);
}

} // namespace genesis::grad

#endif // GENESIS_GRAD_TAPE_H