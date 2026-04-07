--- START OF FILE core/templates/map.h ---

#ifndef MAP_H
#define MAP_H

#include "core/os/memory.h"
#include "core/typedefs.h"
#include "src/big_int_core.h"
#include "src/fixed_math_core.h"

/**
 * Map Template (Red-Black Tree)
 * 
 * An ordered associative container for the Universal Solver.
 * Uses BigIntCore for size to support galactic-scale data sets.
 * Guaranteed deterministic iteration order for cross-client synchronization.
 */
template <typename K, typename V>
class Map {
	enum Color {
		RED,
		BLACK
	};

	struct Element {
		Element *left = nullptr;
		Element *right = nullptr;
		Element *parent = nullptr;
		Color color = RED;
		K key;
		V value;

		Element() {}
		Element(const K &p_key, const V &p_value) :
				key(p_key), value(p_value) {}
	};

	Element *_root = nullptr;
	BigIntCore _size;

	void _rotate_left(Element *p_node) {
		Element *r = p_node->right;
		p_node->right = r->left;
		if (r->left) r->left->parent = p_node;
		r->parent = p_node->parent;
		if (!p_node->parent) _root = r;
		else if (p_node == p_node->parent->left) p_node->parent->left = r;
		else p_node->parent->right = r;
		r->left = p_node;
		p_node->parent = r;
	}

	void _rotate_right(Element *p_node) {
		Element *l = p_node->left;
		p_node->left = l->right;
		if (l->right) l->right->parent = p_node;
		l->parent = p_node->parent;
		if (!p_node->parent) _root = l;
		else if (p_node == p_node->parent->right) p_node->parent->right = l;
		else p_node->parent->left = l;
		l->right = p_node;
		p_node->parent = l;
	}

	void _insert_fixup(Element *p_node) {
		while (p_node->parent && p_node->parent->color == RED) {
			if (p_node->parent == p_node->parent->parent->left) {
				Element *y = p_node->parent->parent->right;
				if (y && y->color == RED) {
					p_node->parent->color = BLACK;
					y->color = BLACK;
					p_node->parent->parent->color = RED;
					p_node = p_node->parent->parent;
				} else {
					if (p_node == p_node->parent->right) {
						p_node = p_node->parent;
						_rotate_left(p_node);
					}
					p_node->parent->color = BLACK;
					p_node->parent->parent->color = RED;
					_rotate_right(p_node->parent->parent);
				}
			} else {
				Element *y = p_node->parent->parent->left;
				if (y && y->color == RED) {
					p_node->parent->color = BLACK;
					y->color = BLACK;
					p_node->parent->parent->color = RED;
					p_node = p_node->parent->parent;
				} else {
					if (p_node == p_node->parent->left) {
						p_node = p_node->parent;
						_rotate_right(p_node);
					}
					p_node->parent->color = BLACK;
					p_node->parent->parent->color = RED;
					_rotate_left(p_node->parent->parent);
				}
			}
		}
		_root->color = BLACK;
	}

	void _cleanup(Element *p_node) {
		if (!p_node) return;
		_cleanup(p_node->left);
		_cleanup(p_node->right);
		memdelete(p_node);
	}

public:
	class Iterator {
		Element *_node = nullptr;
		friend class Map<K, V>;

	public:
		_FORCE_INLINE_ bool is_valid() const { return _node != nullptr; }
		_FORCE_INLINE_ const K &key() const { return _node->key; }
		_FORCE_INLINE_ V &value() { return _node->value; }
		_FORCE_INLINE_ const V &value() const { return _node->value; }

		_FORCE_INLINE_ void next() {
			if (!_node) return;
			if (_node->right) {
				_node = _node->right;
				while (_node->left) _node = _node->left;
			} else {
				Element *p = _node->parent;
				while (p && _node == p->right) {
					_node = p;
					p = p->parent;
				}
				_node = p;
			}
		}
	};

	_FORCE_INLINE_ BigIntCore size() const { return _size; }
	_FORCE_INLINE_ bool is_empty() const { return _root == nullptr; }

	Iterator insert(const K &p_key, const V &p_value) {
		Element *z = memnew(Element(p_key, p_value));
		Element *y = nullptr;
		Element *x = _root;

		while (x) {
			y = x;
			if (z->key < x->key) x = x->left;
			else if (x->key < z->key) x = x->right;
			else {
				x->value = p_value;
				memdelete(z);
				Iterator it; it._node = x;
				return it;
			}
		}

		z->parent = y;
		if (!y) _root = z;
		else if (z->key < y->key) y->left = z;
		else y->right = z;

		_insert_fixup(z);
		_size += BigIntCore(1LL);
		Iterator it; it._node = z;
		return it;
	}

	_FORCE_INLINE_ bool has(const K &p_key) const {
		Element *x = _root;
		while (x) {
			if (p_key < x->key) x = x->left;
			else if (x->key < p_key) x = x->right;
			else return true;
		}
		return false;
	}

	_FORCE_INLINE_ V &operator[](const K &p_key) {
		Element *x = _root;
		while (x) {
			if (p_key < x->key) x = x->left;
			else if (x->key < p_key) x = x->right;
			else return x->value;
		}
		return insert(p_key, V()).value();
	}

	void clear() {
		_cleanup(_root);
		_root = nullptr;
		_size = BigIntCore(0LL);
	}

	Map() : _size(0LL) {}
	~Map() { clear(); }
};

#endif // MAP_H

--- END OF FILE core/templates/map.h ---
