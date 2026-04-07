--- START OF FILE core/object/undo_redo.cpp ---

#include "core/object/undo_redo.h"
#include "core/object/class_db.h"
#include "core/os/os.h"

void UndoRedo::_bind_methods() {
	ClassDB::bind_method(D_METHOD("create_action", "name", "merge_mode"), &UndoRedo::create_action, DEFVAL(MERGE_DISABLE));
	ClassDB::bind_method(D_METHOD("commit_action", "execute"), &UndoRedo::commit_action, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("add_do_method", "object", "method"), &UndoRedo::add_do_method);
	ClassDB::bind_method(D_METHOD("add_undo_method", "object", "method"), &UndoRedo::add_undo_method);
	ClassDB::bind_method(D_METHOD("add_do_property", "object", "property", "value"), &UndoRedo::add_do_property);
	ClassDB::bind_method(D_METHOD("add_undo_property", "object", "property", "value"), &UndoRedo::add_undo_property);
	ClassDB::bind_method(D_METHOD("undo"), &UndoRedo::undo);
	ClassDB::bind_method(D_METHOD("redo"), &UndoRedo::redo);
	ClassDB::bind_method(D_METHOD("clear_history", "increase_version"), &UndoRedo::clear_history, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("get_version"), &UndoRedo::get_version);

	BIND_ENUM_CONSTANT(MERGE_DISABLE);
	BIND_ENUM_CONSTANT(MERGE_ENDS);
	BIND_ENUM_CONSTANT(MERGE_ALL);
}

void UndoRedo::_pop_history_tail() {
	Action *a = history[static_cast<uint32_t>(std::stoll(history.size().to_string())) - 1];
	history.remove_at(history.size() - BigIntCore(1LL));
	memdelete(a);
}

void UndoRedo::_process_operation_list(const List<Operation> &p_ops) {
	for (const typename List<Operation>::Element *E = p_ops.front(); E; E = E->next()) {
		const Operation &op = E->get();
		Callable::CallError ce;
		int argc = op.args.size();
		const Variant **argptr = (const Variant **)alloca(sizeof(Variant *) * argc);
		for (int i = 0; i < argc; i++) {
			argptr[i] = &op.args[i];
		}
		op.object->callp(op.method, argptr, argc, ce);
	}
}

void UndoRedo::create_action(const String &p_name, MergeMode p_mode) {
	if (current_action) {
		memdelete(current_action);
	}
	current_action = memnew(Action);
	current_action->name = p_name;
	current_action->merge_mode = p_mode;
	current_action->timestamp = OS::get_singleton()->get_ticks_usec();
}

void UndoRedo::commit_action(bool p_execute) {
	ERR_FAIL_NULL(current_action);
	committing = true;

	if (p_execute) {
		_process_operation_list(current_action->do_ops);
	}

	// Prune redo history using BigInt indices
	while (BigIntCore(static_cast<int64_t>(history.size())) > history_pos + BigIntCore(1LL)) {
		_pop_history_tail();
	}

	bool merged = false;
	if (current_action->merge_mode != MERGE_DISABLE && history_pos >= BigIntCore(0LL)) {
		Action *last = history[static_cast<uint32_t>(std::stoll(history_pos.to_string()))];
		if (last->name == current_action->name) {
			if (current_action->merge_mode == MERGE_ALL || current_action->merge_mode == MERGE_ENDS) {
				for (const Operation &E : current_action->do_ops) last->do_ops.push_back(E);
				for (const Operation &E : current_action->undo_ops) last->undo_ops.push_front(E);
				memdelete(current_action);
				current_action = nullptr;
				merged = true;
			}
		}
	}

	if (!merged) {
		history.push_back(current_action);
		history_pos += BigIntCore(1LL);
		current_action = nullptr;
	}

	version += BigIntCore(1LL);
	committing = false;
}

bool UndoRedo::undo() {
	if (history_pos < BigIntCore(0LL)) return false;
	Action *a = history[static_cast<uint32_t>(std::stoll(history_pos.to_string()))];
	_process_operation_list(a->undo_ops);
	history_pos -= BigIntCore(1LL);
	version -= BigIntCore(1LL);
	return true;
}

bool UndoRedo::redo() {
	if (history_pos + BigIntCore(1LL) >= BigIntCore(static_cast<int64_t>(history.size()))) return false;
	history_pos += BigIntCore(1LL);
	Action *a = history[static_cast<uint32_t>(std::stoll(history_pos.to_string()))];
	_process_operation_list(a->do_ops);
	version += BigIntCore(1LL);
	return true;
}

void UndoRedo::clear_history(bool p_increase_version) {
	while (history.size() > 0) {
		_pop_history_tail();
	}
	history_pos = BigIntCore(-1LL);
	if (p_increase_version) {
		version += BigIntCore(1LL);
	}
}

BigIntCore UndoRedo::get_history_count() const { return BigIntCore(static_cast<int64_t>(history.size())); }
BigIntCore UndoRedo::get_version() const { return version; }

UndoRedo::UndoRedo() {
	history_pos = BigIntCore(-1LL);
	version = BigIntCore(0LL);
}

UndoRedo::~UndoRedo() {
	clear_history(false);
}

--- END OF FILE core/object/undo_redo.cpp ---
