--- START OF FILE core/object/undo_redo.h ---

#ifndef UNDO_REDO_H
#define UNDO_REDO_H

#include "core/object/object.h"
#include "core/templates/vector.h"
#include "core/templates/list.h"
#include "core/variant/variant.h"
#include "src/big_int_core.h"

/**
 * UndoRedo
 * 
 * Manages an action history stack for deterministic state restoration.
 * Uses BigIntCore for indexing and versioning to prevent overflow in 
 * massive-scale simulation sessions.
 * Optimized for zero-copy storage of high-precision math components.
 */
class UndoRedo : public Object {
	GDCLASS(UndoRedo, Object);

public:
	enum MergeMode {
		MERGE_DISABLE,
		MERGE_ENDS,
		MERGE_ALL
	};

	struct Operation {
		Object *object = nullptr;
		StringName method;
		Vector<Variant> args;
	};

	struct Action {
		String name;
		List<Operation> do_ops;
		List<Operation> undo_ops;
		MergeMode merge_mode = MERGE_DISABLE;
		BigIntCore timestamp;
	};

private:
	Vector<Action *> history;
	BigIntCore history_pos;
	BigIntCore version;
	Action *current_action = nullptr;
	bool committing = false;

	void _pop_history_tail();
	void _process_operation_list(const List<Operation> &p_ops);

protected:
	static void _bind_methods();

public:
	// ------------------------------------------------------------------------
	// Transaction API
	// ------------------------------------------------------------------------

	/**
	 * create_action()
	 * Starts a new atomic action. Uses BigIntCore for the internal 
	 * simulation clock timestamping.
	 */
	void create_action(const String &p_name, MergeMode p_mode = MERGE_DISABLE);

	/**
	 * commit_action()
	 * Finalizes and executes the current action. 
	 * Prunes the redo branch using BigIntCore indices.
	 */
	void commit_action(bool p_execute = true);

	bool is_committing() const { return committing; }

	// ------------------------------------------------------------------------
	// Operation API
	// ------------------------------------------------------------------------

	void add_do_method(Object *p_object, const StringName &p_method, const Variant &p_arg1 = Variant(), const Variant &p_arg2 = Variant(), const Variant &p_arg3 = Variant(), const Variant &p_arg4 = Variant(), const Variant &p_arg5 = Variant());
	void add_undo_method(Object *p_object, const StringName &p_method, const Variant &p_arg1 = Variant(), const Variant &p_arg2 = Variant(), const Variant &p_arg3 = Variant(), const Variant &p_arg4 = Variant(), const Variant &p_arg5 = Variant());

	void add_do_property(Object *p_object, const StringName &p_property, const Variant &p_value);
	void add_undo_property(Object *p_object, const StringName &p_property, const Variant &p_value);

	// ------------------------------------------------------------------------
	// Execution API
	// ------------------------------------------------------------------------

	bool undo();
	bool redo();

	void clear_history(bool p_increase_version = true);

	// ------------------------------------------------------------------------
	// State & Telemetry
	// ------------------------------------------------------------------------

	BigIntCore get_history_count() const;
	BigIntCore get_version() const;
	String get_action_name(const BigIntCore &p_idx) const;

	UndoRedo();
	~UndoRedo();
};

#endif // UNDO_REDO_H

--- END OF FILE core/object/undo_redo.h ---
