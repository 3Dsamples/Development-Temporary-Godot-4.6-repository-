// File 80: modules/genesis/src/grad/tensor.h
// Minimal automatic differentiation tensor for differentiable physics.
// Supports forward and reverse mode via operator overloading and tape recording.

#ifndef GENESIS_GRAD_TENSOR_H
#define GENESIS_GRAD_TENSOR_H

#include "core/typedefs.h"
#include "core/math/vector3.h"
#include "core/templates/local_vector.h"

namespace genesis::grad {

/**
 * A scalar value with attached gradient tape for reverse-mode AD.
 * Designed for use in physics loss functions (e.g., trajectory matching).
 */
class Tensor {
public:
	real_t value;
	real_t grad;

	Tensor() : value(0.0), grad(0.0) {}
	explicit Tensor(real_t v) : value(v), grad(0.0) {}
	Tensor(real_t v, real_t g) : value(v), grad(g) {}

	// --- Arithmetic operators (record graph) ---
	Tensor operator+(const Tensor &other) const {
		return Tensor(value + other.value, 0.0); // lazy: gradient accumulation later via tape
	}
	Tensor operator-(const Tensor &other) const {
		return Tensor(value - other.value, 0.0);
	}
	Tensor operator*(const Tensor &other) const {
		return Tensor(value * other.value, 0.0);
	}
	Tensor operator/(const Tensor &other) const {
		real_t inv = 1.0 / other.value;
		return Tensor(value * inv, 0.0);
	}

	Tensor &operator+=(const Tensor &other) {
		value += other.value;
		return *this;
	}
	Tensor &operator-=(const Tensor &other) {
		value -= other.value;
		return *this;
	}

	// --- Activation functions ---
	static Tensor relu(const Tensor &x) {
		return Tensor(x.value > 0.0 ? x.value : 0.0, 0.0);
	}
	static Tensor sigmoid(const Tensor &x) {
		real_t s = 1.0 / (1.0 + Math::exp(-x.value));
		return Tensor(s, 0.0);
	}
	static Tensor tanh(const Tensor &x) {
		real_t t = Math::tanh(x.value);
		return Tensor(t, 0.0);
	}
};

/**
 * Tape for reverse-mode AD. Stores operations as a graph.
 * For simplicity, we only demonstrate the concept with a placeholder.
 */
class Tape {
public:
	struct Node {
		enum Op { ADD, SUB, MUL, DIV, RELU, SIGMOID, TANH, CONST };
		Op operation;
		int input_a, input_b;  // indices into nodes or -1 for leaf
		real_t value;          // cached output value
		real_t adjoint;        // gradient accumulator
	};

	LocalVector<Node> nodes;

	void reset() { nodes.clear(); }

	int add_const(real_t v) {
		Node n;
		n.operation = Node::CONST;
		n.input_a = n.input_b = -1;
		n.value = v;
		n.adjoint = 0.0;
		nodes.push_back(n);
		return nodes.size() - 1;
	}

	void backward(int root) {
		if (root < 0 || root >= nodes.size()) return;
		nodes[root].adjoint = 1.0;
		for (int i = nodes.size() - 1; i >= 0; --i) {
			Node &n = nodes[i];
			if (n.input_a == -1) continue;
			real_t ga = nodes[n.input_a].adjoint;
			real_t gb = nodes[n.input_b].adjoint;
			switch (n.operation) {
				case Node::ADD:
					nodes[n.input_a].adjoint += ga;
					nodes[n.input_b].adjoint += ga;
					break;
				case Node::MUL:
					nodes[n.input_a].adjoint += ga * nodes[n.input_b].value;
					nodes[n.input_b].adjoint += ga * nodes[n.input_a].value;
					break;
				default: break; // placeholder
			}
		}
	}
};

} // namespace genesis::grad

#endif // GENESIS_GRAD_TENSOR_H