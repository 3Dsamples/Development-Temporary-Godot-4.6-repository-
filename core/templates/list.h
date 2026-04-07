--- START OF FILE core/templates/list.h ---

#ifndef LIST_H
#define LIST_H

#include "core/typedefs.h"
#include "core/os/memory.h"
#include "src/big_int_core.h"

/**
 * List Template
 * 
 * A doubly-linked list for managing simulation entities and observers.
 * Uses BigIntCore for size tracking to maintain galactic-scale compatibility.
 * Node allocation is handled via the engine's centralized memory system.
 */
template <typename T>
class List {
public:
	struct Element {
	private:
		friend class List<T>;
		T value;
		Element *next_ptr = nullptr;
		Element *prev_ptr = nullptr;

	public:
		_FORCE_INLINE_ const T &get() const { return value; }
		_FORCE_INLINE_ T &get() { return value; }
		_FORCE_INLINE_ Element *next() { return next_ptr; }
		_FORCE_INLINE_ const Element *next() const { return next_ptr; }
		_FORCE_INLINE_ Element *prev() { return prev_ptr; }
		_FORCE_INLINE_ const Element *prev() const { return prev_ptr; }

		Element() {}
	};

private:
	Element *first_ptr = nullptr;
	Element *last_ptr = nullptr;
	BigIntCore _size;

public:
	// ------------------------------------------------------------------------
	// Accessors
	// ------------------------------------------------------------------------

	_FORCE_INLINE_ Element *front() { return first_ptr; }
	_FORCE_INLINE_ const Element *front() const { return first_ptr; }
	_FORCE_INLINE_ Element *back() { return last_ptr; }
	_FORCE_INLINE_ const Element *back() const { return last_ptr; }

	_FORCE_INLINE_ BigIntCore size() const { return _size; }
	_FORCE_INLINE_ bool is_empty() const { return first_ptr == nullptr; }

	// ------------------------------------------------------------------------
	// Modification API
	// ------------------------------------------------------------------------

	/**
	 * push_back()
	 * Adds an element to the end of the list. O(1) complexity.
	 */
	Element *push_back(const T &p_value) {
		Element *e = memnew(Element);
		e->value = p_value;
		e->prev_ptr = last_ptr;
		e->next_ptr = nullptr;

		if (last_ptr) {
			last_ptr->next_ptr = e;
		} else {
			first_ptr = e;
		}

		last_ptr = e;
		_size += BigIntCore(1LL);
		return e;
	}

	/**
	 * push_front()
	 * Adds an element to the start of the list. O(1) complexity.
	 */
	Element *push_front(const T &p_value) {
		Element *e = memnew(Element);
		e->value = p_value;
		e->prev_ptr = nullptr;
		e->next_ptr = first_ptr;

		if (first_ptr) {
			first_ptr->prev_ptr = e;
		} else {
			last_ptr = e;
		}

		first_ptr = e;
		_size += BigIntCore(1LL);
		return e;
	}

	/**
	 * erase()
	 * Removes a specific element from the list. O(1) complexity.
	 */
	void erase(Element *p_element) {
		if (unlikely(!p_element)) return;

		if (p_element->prev_ptr) {
			p_element->prev_ptr->next_ptr = p_element->next_ptr;
		} else {
			first_ptr = p_element->next_ptr;
		}

		if (p_element->next_ptr) {
			p_element->next_ptr->prev_ptr = p_element->prev_ptr;
		} else {
			last_ptr = p_element->prev_ptr;
		}

		memdelete(p_element);
		_size -= BigIntCore(1LL);
	}

	void clear() {
		while (first_ptr) {
			erase(first_ptr);
		}
	}

	// ------------------------------------------------------------------------
	// Lifecycle
	// ------------------------------------------------------------------------

	List() : _size(0LL) {}
	~List() { clear(); }

	void operator=(const List &p_from) {
		clear();
		for (const Element *E = p_from.front(); E; E = E->next()) {
			push_back(E->get());
		}
	}
};

#endif // LIST_H

--- END OF FILE core/templates/list.h ---
