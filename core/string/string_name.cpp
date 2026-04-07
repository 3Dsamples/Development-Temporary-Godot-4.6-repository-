--- START OF FILE core/string/string_name.cpp ---

#include "core/string/string_name.h"
#include "core/templates/hash_map.h"
#include "core/os/memory.h"
#include <mutex>

/**
 * Global StringName Pool
 * 
 * ETEngine Strategy: Uses a thread-safe hash map to store unique string data.
 * This ensures that every unique string exists only once in memory.
 */
static std::mutex *string_name_mutex = nullptr;
static HashMap<String, StringName::_Data *> *string_name_pool = nullptr;

StringName::StringName(const StringName &p_from) {
	_data = p_from._data;
	if (_data) {
		_data->refcount.ref();
	}
}

StringName::StringName(const String &p_string) {
	if (p_string.is_empty()) {
		return;
	}

	std::lock_guard<std::mutex> lock(*string_name_mutex);

	if (string_name_pool->has(p_string)) {
		_data = (*string_name_pool)[p_string];
		_data->refcount.ref();
		return;
	}

	_data = memnew(_Data);
	_data->refcount.init();
	_data->string = p_string;
	_data->hash = p_string.hash();
	
	string_name_pool->insert(p_string, _data);
}

StringName::StringName(const char *p_contents) : StringName(String(p_contents)) {}

StringName::~StringName() {
	if (!_data) {
		return;
	}

	std::lock_guard<std::mutex> lock(*string_name_mutex);

	if (_data->refcount.unref()) {
		string_name_pool->erase(_data->string);
		memdelete(_data);
	}
}

void StringName::operator=(const StringName &p_from) {
	if (_data == p_from._data) {
		return;
	}

	if (_data) {
		std::lock_guard<std::mutex> lock(*string_name_mutex);
		if (_data->refcount.unref()) {
			string_name_pool->erase(_data->string);
			memdelete(_data);
		}
	}

	_data = p_from._data;
	if (_data) {
		_data->refcount.ref();
	}
}

StringName::operator String() const {
	if (_data) {
		return _data->string;
	}
	return String();
}

void StringName::setup() {
	string_name_mutex = new std::mutex();
	string_name_pool = new HashMap<String, StringName::_Data *>();
}

void StringName::cleanup() {
	delete string_name_pool;
	string_name_pool = nullptr;
	delete string_name_mutex;
	string_name_mutex = nullptr;
}

--- END OF FILE core/string/string_name.cpp ---
