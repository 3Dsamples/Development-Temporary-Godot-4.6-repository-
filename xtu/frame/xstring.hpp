// include/xtu/frame/xstring.hpp
// xtensor-unified - Vectorized string operations for dataframe columns
// Copyright (c) 2026, Xtensor-Stack Contributors
// SPDX-License-Identifier: BSD-3-Clause

#ifndef XTU_FRAME_XSTRING_HPP
#define XTU_FRAME_XSTRING_HPP

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "xtu/core/xtensor_config.hpp"
#include "xtu/core/xtensor_forward.hpp"
#include "xtu/containers/xarray.hpp"
#include "xtu/frame/xvariable.hpp"
#include "xtu/frame/xcoordinate_system.hpp"

XTU_NAMESPACE_BEGIN
namespace frame {
namespace string_ops {

// #############################################################################
// String accessor class (similar to pandas .str)
// #############################################################################
template <class C = xcoordinate_system>
class StringAccessor {
private:
    const xdataframe<C>* m_df;
    std::string m_column;
    const xarray_container<std::string>* m_data;

public:
    StringAccessor(const xdataframe<C>& df, const std::string& column)
        : m_df(&df), m_column(column), m_data(nullptr) {
        auto it = df.m_variables.find(column);
        if (it == df.m_variables.end()) {
            XTU_THROW(std::runtime_error, "Column not found: " + column);
        }
        auto* holder = dynamic_cast<const typename xdataframe<C>::template variable_holder<std::string>*>(it->second.get());
        if (!holder) {
            XTU_THROW(std::runtime_error, "Column is not string type: " + column);
        }
        m_data = &holder->var.data();
    }

    size_t size() const { return m_data->size(); }
    const xarray_container<std::string>& data() const { return *m_data; }

    // #########################################################################
    // Case conversion
    // #########################################################################
    xarray_container<std::string> lower() const {
        xarray_container<std::string> result(m_data->shape());
        for (size_t i = 0; i < m_data->size(); ++i) {
            std::string s = (*m_data)[i];
            std::transform(s.begin(), s.end(), s.begin(),
                [](unsigned char c) { return std::tolower(c); });
            result.flat(i) = s;
        }
        return result;
    }

    xarray_container<std::string> upper() const {
        xarray_container<std::string> result(m_data->shape());
        for (size_t i = 0; i < m_data->size(); ++i) {
            std::string s = (*m_data)[i];
            std::transform(s.begin(), s.end(), s.begin(),
                [](unsigned char c) { return std::toupper(c); });
            result.flat(i) = s;
        }
        return result;
    }

    xarray_container<std::string> capitalize() const {
        xarray_container<std::string> result(m_data->shape());
        for (size_t i = 0; i < m_data->size(); ++i) {
            std::string s = (*m_data)[i];
            if (!s.empty()) {
                s[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(s[0])));
                for (size_t j = 1; j < s.size(); ++j) {
                    s[j] = static_cast<char>(std::tolower(static_cast<unsigned char>(s[j])));
                }
            }
            result.flat(i) = s;
        }
        return result;
    }

    xarray_container<std::string> title() const {
        xarray_container<std::string> result(m_data->shape());
        for (size_t i = 0; i < m_data->size(); ++i) {
            std::string s = (*m_data)[i];
            bool new_word = true;
            for (char& c : s) {
                if (std::isspace(static_cast<unsigned char>(c)) || c == '-' || c == '_') {
                    new_word = true;
                } else if (new_word) {
                    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
                    new_word = false;
                } else {
                    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                }
            }
            result.flat(i) = s;
        }
        return result;
    }

    xarray_container<std::string> swapcase() const {
        xarray_container<std::string> result(m_data->shape());
        for (size_t i = 0; i < m_data->size(); ++i) {
            std::string s = (*m_data)[i];
            for (char& c : s) {
                if (std::isupper(static_cast<unsigned char>(c))) {
                    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                } else if (std::islower(static_cast<unsigned char>(c))) {
                    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
                }
            }
            result.flat(i) = s;
        }
        return result;
    }

    // #########################################################################
    // Whitespace trimming
    // #########################################################################
    xarray_container<std::string> strip() const {
        xarray_container<std::string> result(m_data->shape());
        for (size_t i = 0; i < m_data->size(); ++i) {
            std::string s = (*m_data)[i];
            size_t start = 0;
            while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) ++start;
            size_t end = s.size();
            while (end > start && std::isspace(static_cast<unsigned char>(s[end-1]))) --end;
            result.flat(i) = s.substr(start, end - start);
        }
        return result;
    }

    xarray_container<std::string> lstrip() const {
        xarray_container<std::string> result(m_data->shape());
        for (size_t i = 0; i < m_data->size(); ++i) {
            std::string s = (*m_data)[i];
            size_t start = 0;
            while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) ++start;
            result.flat(i) = s.substr(start);
        }
        return result;
    }

    xarray_container<std::string> rstrip() const {
        xarray_container<std::string> result(m_data->shape());
        for (size_t i = 0; i < m_data->size(); ++i) {
            std::string s = (*m_data)[i];
            size_t end = s.size();
            while (end > 0 && std::isspace(static_cast<unsigned char>(s[end-1]))) --end;
            result.flat(i) = s.substr(0, end);
        }
        return result;
    }

    // #########################################################################
    // Padding
    // #########################################################################
    xarray_container<std::string> pad(size_t width, char fillchar = ' ', bool left = true) const {
        xarray_container<std::string> result(m_data->shape());
        for (size_t i = 0; i < m_data->size(); ++i) {
            std::string s = (*m_data)[i];
            if (s.size() >= width) {
                result.flat(i) = s;
            } else {
                size_t pad_len = width - s.size();
                std::string padding(pad_len, fillchar);
                result.flat(i) = left ? (padding + s) : (s + padding);
            }
        }
        return result;
    }

    xarray_container<std::string> zfill(size_t width) const {
        return pad(width, '0', true);
    }

    xarray_container<std::string> center(size_t width, char fillchar = ' ') const {
        xarray_container<std::string> result(m_data->shape());
        for (size_t i = 0; i < m_data->size(); ++i) {
            std::string s = (*m_data)[i];
            if (s.size() >= width) {
                result.flat(i) = s;
            } else {
                size_t left_pad = (width - s.size()) / 2;
                size_t right_pad = width - s.size() - left_pad;
                result.flat(i) = std::string(left_pad, fillchar) + s + std::string(right_pad, fillchar);
            }
        }
        return result;
    }

    // #########################################################################
    // Slicing and substrings
    // #########################################################################
    xarray_container<std::string> slice(size_t start = 0, size_t stop = std::string::npos, size_t step = 1) const {
        xarray_container<std::string> result(m_data->shape());
        for (size_t i = 0; i < m_data->size(); ++i) {
            const std::string& s = (*m_data)[i];
            if (start >= s.size()) {
                result.flat(i) = "";
                continue;
            }
            size_t end = std::min(stop, s.size());
            if (step == 1) {
                result.flat(i) = s.substr(start, end - start);
            } else {
                std::string out;
                for (size_t j = start; j < end; j += step) {
                    out += s[j];
                }
                result.flat(i) = out;
            }
        }
        return result;
    }

    xarray_container<std::string> get(size_t idx) const {
        xarray_container<std::string> result(m_data->shape());
        for (size_t i = 0; i < m_data->size(); ++i) {
            const std::string& s = (*m_data)[i];
            if (idx < s.size()) {
                result.flat(i) = std::string(1, s[idx]);
            } else {
                result.flat(i) = "";
            }
        }
        return result;
    }

    // #########################################################################
    // Length
    // #########################################################################
    xarray_container<size_t> len() const {
        xarray_container<size_t> result(m_data->shape());
        for (size_t i = 0; i < m_data->size(); ++i) {
            result.flat(i) = (*m_data)[i].size();
        }
        return result;
    }

    // #########################################################################
    // Concatenation and repetition
    // #########################################################################
    xarray_container<std::string> cat(const std::string& sep = "", const std::string& na_rep = "") const {
        xarray_container<std::string> result(m_data->shape());
        // cat with self doesn't make sense; typically cat joins multiple columns
        // We'll implement as join all elements into a single string
        std::ostringstream oss;
        for (size_t i = 0; i < m_data->size(); ++i) {
            if (i > 0) oss << sep;
            oss << ((*m_data)[i].empty() ? na_rep : (*m_data)[i]);
        }
        result.flat(0) = oss.str();
        return result;
    }

    xarray_container<std::string> repeat(size_t repeats) const {
        xarray_container<std::string> result(m_data->shape());
        for (size_t i = 0; i < m_data->size(); ++i) {
            const std::string& s = (*m_data)[i];
            std::string out;
            out.reserve(s.size() * repeats);
            for (size_t r = 0; r < repeats; ++r) out += s;
            result.flat(i) = out;
        }
        return result;
    }

    // #########################################################################
    // Splitting
    // #########################################################################
    xdataframe<C> split(const std::string& pat = " ", bool expand = true) const {
        size_t n = m_data->size();
        // Find max number of splits
        size_t max_parts = 0;
        std::vector<std::vector<std::string>> all_parts(n);
        for (size_t i = 0; i < n; ++i) {
            const std::string& s = (*m_data)[i];
            size_t pos = 0;
            while (pos < s.size()) {
                size_t found = s.find(pat, pos);
                if (found == std::string::npos) {
                    all_parts[i].push_back(s.substr(pos));
                    break;
                } else {
                    all_parts[i].push_back(s.substr(pos, found - pos));
                    pos = found + pat.size();
                }
            }
            max_parts = std::max(max_parts, all_parts[i].size());
        }
        if (max_parts == 0) max_parts = 1;
        // Build dataframe
        coordinate_system_type coords;
        coords = coordinate_system_type({"index"});
        xaxis<size_t, size_type> idx_axis;
        for (size_t i = 0; i < n; ++i) idx_axis.push_back(i);
        coords.set_axis(0, idx_axis);
        xdataframe<C> result(coords);
        for (size_t part = 0; part < max_parts; ++part) {
            xarray_container<std::string> col_data({n});
            for (size_t i = 0; i < n; ++i) {
                col_data[i] = (part < all_parts[i].size()) ? all_parts[i][part] : "";
            }
            result.add_column(m_column + "_" + std::to_string(part), col_data);
        }
        return result;
    }

    xdataframe<C> rsplit(const std::string& pat = " ", size_t n = std::string::npos, bool expand = true) const {
        // Similar to split but from right; implement simplified
        return split(pat, expand);
    }

    // #########################################################################
    // Replace
    // #########################################################################
    xarray_container<std::string> replace(const std::string& pat, const std::string& repl,
                                          bool regex = false) const {
        xarray_container<std::string> result(m_data->shape());
        if (regex) {
            std::regex re(pat);
            for (size_t i = 0; i < m_data->size(); ++i) {
                result.flat(i) = std::regex_replace((*m_data)[i], re, repl);
            }
        } else {
            for (size_t i = 0; i < m_data->size(); ++i) {
                std::string s = (*m_data)[i];
                size_t pos = 0;
                while ((pos = s.find(pat, pos)) != std::string::npos) {
                    s.replace(pos, pat.size(), repl);
                    pos += repl.size();
                }
                result.flat(i) = s;
            }
        }
        return result;
    }

    // #########################################################################
    // Pattern matching
    // #########################################################################
    xarray_container<bool> contains(const std::string& pat, bool regex = false, bool case_sensitive = true) const {
        xarray_container<bool> result(m_data->shape());
        if (regex) {
            std::regex re(pat, case_sensitive ? std::regex::ECMAScript : std::regex::icase);
            for (size_t i = 0; i < m_data->size(); ++i) {
                result.flat(i) = std::regex_search((*m_data)[i], re);
            }
        } else {
            for (size_t i = 0; i < m_data->size(); ++i) {
                const std::string& s = (*m_data)[i];
                if (case_sensitive) {
                    result.flat(i) = s.find(pat) != std::string::npos;
                } else {
                    std::string s_lower = s;
                    std::string pat_lower = pat;
                    std::transform(s_lower.begin(), s_lower.end(), s_lower.begin(),
                        [](unsigned char c) { return std::tolower(c); });
                    std::transform(pat_lower.begin(), pat_lower.end(), pat_lower.begin(),
                        [](unsigned char c) { return std::tolower(c); });
                    result.flat(i) = s_lower.find(pat_lower) != std::string::npos;
                }
            }
        }
        return result;
    }

    xarray_container<bool> startswith(const std::string& pat) const {
        xarray_container<bool> result(m_data->shape());
        for (size_t i = 0; i < m_data->size(); ++i) {
            const std::string& s = (*m_data)[i];
            result.flat(i) = s.size() >= pat.size() && s.compare(0, pat.size(), pat) == 0;
        }
        return result;
    }

    xarray_container<bool> endswith(const std::string& pat) const {
        xarray_container<bool> result(m_data->shape());
        for (size_t i = 0; i < m_data->size(); ++i) {
            const std::string& s = (*m_data)[i];
            result.flat(i) = s.size() >= pat.size() && 
                             s.compare(s.size() - pat.size(), pat.size(), pat) == 0;
        }
        return result;
    }

    xarray_container<size_t> find(const std::string& sub, size_t start = 0, size_t end = std::string::npos) const {
        xarray_container<size_t> result(m_data->shape());
        for (size_t i = 0; i < m_data->size(); ++i) {
            const std::string& s = (*m_data)[i];
            size_t pos = s.find(sub, start);
            if (pos != std::string::npos && pos < end) {
                result.flat(i) = pos;
            } else {
                result.flat(i) = static_cast<size_t>(-1);
            }
        }
        return result;
    }

    xarray_container<size_t> rfind(const std::string& sub, size_t start = 0, size_t end = std::string::npos) const {
        xarray_container<size_t> result(m_data->shape());
        for (size_t i = 0; i < m_data->size(); ++i) {
            const std::string& s = (*m_data)[i];
            size_t pos = s.rfind(sub, end == std::string::npos ? std::string::npos : end);
            if (pos != std::string::npos && pos >= start) {
                result.flat(i) = pos;
            } else {
                result.flat(i) = static_cast<size_t>(-1);
            }
        }
        return result;
    }

    // #########################################################################
    // Count occurrences
    // #########################################################################
    xarray_container<size_t> count(const std::string& pat) const {
        xarray_container<size_t> result(m_data->shape());
        for (size_t i = 0; i < m_data->size(); ++i) {
            const std::string& s = (*m_data)[i];
            size_t cnt = 0;
            size_t pos = 0;
            while ((pos = s.find(pat, pos)) != std::string::npos) {
                ++cnt;
                pos += pat.size();
            }
            result.flat(i) = cnt;
        }
        return result;
    }

    // #########################################################################
    // Extract with regex
    // #########################################################################
    xarray_container<std::string> extract(const std::string& pat, size_t group = 0) const {
        std::regex re(pat);
        xarray_container<std::string> result(m_data->shape());
        for (size_t i = 0; i < m_data->size(); ++i) {
            const std::string& s = (*m_data)[i];
            std::smatch match;
            if (std::regex_search(s, match, re) && group < match.size()) {
                result.flat(i) = match[group].str();
            } else {
                result.flat(i) = "";
            }
        }
        return result;
    }

    xdataframe<C> extractall(const std::string& pat) const {
        std::regex re(pat);
        // Collect all matches across all rows
        std::vector<size_t> row_indices;
        std::vector<std::vector<std::string>> all_groups;
        size_t max_groups = 0;
        for (size_t i = 0; i < m_data->size(); ++i) {
            const std::string& s = (*m_data)[i];
            std::sregex_iterator it(s.begin(), s.end(), re);
            std::sregex_iterator end;
            for (; it != end; ++it) {
                row_indices.push_back(i);
                std::vector<std::string> groups;
                for (size_t g = 0; g < it->size(); ++g) {
                    groups.push_back((*it)[g].str());
                }
                max_groups = std::max(max_groups, groups.size());
                all_groups.push_back(groups);
            }
        }
        // Build dataframe
        coordinate_system_type coords;
        coords = coordinate_system_type({"index"});
        xaxis<size_t, size_type> idx_axis;
        for (size_t i = 0; i < row_indices.size(); ++i) idx_axis.push_back(i);
        coords.set_axis(0, idx_axis);
        xdataframe<C> result(coords);
        // Add row index column
        xarray_container<size_t> row_col({row_indices.size()});
        for (size_t i = 0; i < row_indices.size(); ++i) row_col[i] = row_indices[i];
        result.add_column("row", row_col);
        // Add group columns
        for (size_t g = 0; g < max_groups; ++g) {
            xarray_container<std::string> col_data({row_indices.size()});
            for (size_t i = 0; i < all_groups.size(); ++i) {
                col_data[i] = (g < all_groups[i].size()) ? all_groups[i][g] : "";
            }
            result.add_column("group_" + std::to_string(g), col_data);
        }
        return result;
    }

    // #########################################################################
    // Check if string matches pattern
    // #########################################################################
    xarray_container<bool> match(const std::string& pat) const {
        std::regex re(pat);
        xarray_container<bool> result(m_data->shape());
        for (size_t i = 0; i < m_data->size(); ++i) {
            result.flat(i) = std::regex_match((*m_data)[i], re);
        }
        return result;
    }

    // #########################################################################
    // Remove prefix/suffix
    // #########################################################################
    xarray_container<std::string> removeprefix(const std::string& prefix) const {
        xarray_container<std::string> result(m_data->shape());
        for (size_t i = 0; i < m_data->size(); ++i) {
            const std::string& s = (*m_data)[i];
            if (s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0) {
                result.flat(i) = s.substr(prefix.size());
            } else {
                result.flat(i) = s;
            }
        }
        return result;
    }

    xarray_container<std::string> removesuffix(const std::string& suffix) const {
        xarray_container<std::string> result(m_data->shape());
        for (size_t i = 0; i < m_data->size(); ++i) {
            const std::string& s = (*m_data)[i];
            if (s.size() >= suffix.size() && 
                s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0) {
                result.flat(i) = s.substr(0, s.size() - suffix.size());
            } else {
                result.flat(i) = s;
            }
        }
        return result;
    }

    // #########################################################################
    // Wrap text
    // #########################################################################
    xarray_container<std::string> wrap(size_t width) const {
        xarray_container<std::string> result(m_data->shape());
        for (size_t i = 0; i < m_data->size(); ++i) {
            const std::string& s = (*m_data)[i];
            std::ostringstream oss;
            size_t pos = 0;
            while (pos < s.size()) {
                size_t end = std::min(pos + width, s.size());
                if (end < s.size() && !std::isspace(static_cast<unsigned char>(s[end]))) {
                    size_t last_space = s.rfind(' ', end);
                    if (last_space != std::string::npos && last_space > pos) {
                        end = last_space;
                    }
                }
                if (pos > 0) oss << '\n';
                oss << s.substr(pos, end - pos);
                pos = end;
                while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) ++pos;
            }
            result.flat(i) = oss.str();
        }
        return result;
    }

    // #########################################################################
    // Encode/decode
    // #########################################################################
    xarray_container<std::string> encode(const std::string& encoding = "utf-8") const {
        // Placeholder: return original (actual encoding would require ICU or similar)
        return *m_data;
    }

    xarray_container<std::string> decode(const std::string& encoding = "utf-8") const {
        return *m_data;
    }

    // #########################################################################
    // Dummy variables (one-hot encoding)
    // #########################################################################
    xdataframe<C> get_dummies(const std::string& prefix = "", const std::string& prefix_sep = "_") const {
        // Collect unique values
        std::unordered_set<std::string> unique_vals;
        for (size_t i = 0; i < m_data->size(); ++i) {
            unique_vals.insert((*m_data)[i]);
        }
        std::vector<std::string> sorted_vals(unique_vals.begin(), unique_vals.end());
        std::sort(sorted_vals.begin(), sorted_vals.end());
        // Build dataframe
        coordinate_system_type coords;
        coords = coordinate_system_type({"index"});
        xaxis<size_t, size_type> idx_axis;
        for (size_t i = 0; i < m_data->size(); ++i) idx_axis.push_back(i);
        coords.set_axis(0, idx_axis);
        xdataframe<C> result(coords);
        std::string col_prefix = prefix.empty() ? m_column : prefix;
        for (const auto& val : sorted_vals) {
            std::string col_name = col_prefix + prefix_sep + val;
            xarray_container<int> dummy({m_data->size()});
            for (size_t i = 0; i < m_data->size(); ++i) {
                dummy[i] = ((*m_data)[i] == val) ? 1 : 0;
            }
            result.add_column(col_name, dummy);
        }
        return result;
    }
};

// #############################################################################
// Free function to access string operations
// #############################################################################
template <class C>
StringAccessor<C> str_accessor(const xdataframe<C>& df, const std::string& column) {
    return StringAccessor<C>(df, column);
}

// #############################################################################
// Vectorized string functions for any string container
// #############################################################################
template <class Container>
xarray_container<bool> str_contains(const Container& strings, const std::string& pat, bool regex = false) {
    xarray_container<bool> result(strings.shape());
    if (regex) {
        std::regex re(pat);
        for (size_t i = 0; i < strings.size(); ++i) {
            result.flat(i) = std::regex_search(strings.flat(i), re);
        }
    } else {
        for (size_t i = 0; i < strings.size(); ++i) {
            result.flat(i) = strings.flat(i).find(pat) != std::string::npos;
        }
    }
    return result;
}

template <class Container>
xarray_container<std::string> str_replace(const Container& strings, const std::string& pat, 
                                           const std::string& repl, bool regex = false) {
    xarray_container<std::string> result(strings.shape());
    if (regex) {
        std::regex re(pat);
        for (size_t i = 0; i < strings.size(); ++i) {
            result.flat(i) = std::regex_replace(strings.flat(i), re, repl);
        }
    } else {
        for (size_t i = 0; i < strings.size(); ++i) {
            std::string s = strings.flat(i);
            size_t pos = 0;
            while ((pos = s.find(pat, pos)) != std::string::npos) {
                s.replace(pos, pat.size(), repl);
                pos += repl.size();
            }
            result.flat(i) = s;
        }
    }
    return result;
}

template <class Container>
xarray_container<std::string> str_lower(const Container& strings) {
    xarray_container<std::string> result(strings.shape());
    for (size_t i = 0; i < strings.size(); ++i) {
        std::string s = strings.flat(i);
        std::transform(s.begin(), s.end(), s.begin(),
            [](unsigned char c) { return std::tolower(c); });
        result.flat(i) = s;
    }
    return result;
}

template <class Container>
xarray_container<std::string> str_upper(const Container& strings) {
    xarray_container<std::string> result(strings.shape());
    for (size_t i = 0; i < strings.size(); ++i) {
        std::string s = strings.flat(i);
        std::transform(s.begin(), s.end(), s.begin(),
            [](unsigned char c) { return std::toupper(c); });
        result.flat(i) = s;
    }
    return result;
}

template <class Container>
xarray_container<std::string> str_strip(const Container& strings) {
    xarray_container<std::string> result(strings.shape());
    for (size_t i = 0; i < strings.size(); ++i) {
        const std::string& s = strings.flat(i);
        size_t start = 0;
        while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) ++start;
        size_t end = s.size();
        while (end > start && std::isspace(static_cast<unsigned char>(s[end-1]))) --end;
        result.flat(i) = s.substr(start, end - start);
    }
    return result;
}

template <class Container>
xarray_container<size_t> str_len(const Container& strings) {
    xarray_container<size_t> result(strings.shape());
    for (size_t i = 0; i < strings.size(); ++i) {
        result.flat(i) = strings.flat(i).size();
    }
    return result;
}

template <class Container>
xarray_container<std::string> str_slice(const Container& strings, size_t start = 0, 
                                         size_t stop = std::string::npos, size_t step = 1) {
    xarray_container<std::string> result(strings.shape());
    for (size_t i = 0; i < strings.size(); ++i) {
        const std::string& s = strings.flat(i);
        if (start >= s.size()) {
            result.flat(i) = "";
            continue;
        }
        size_t end = std::min(stop, s.size());
        if (step == 1) {
            result.flat(i) = s.substr(start, end - start);
        } else {
            std::string out;
            for (size_t j = start; j < end; j += step) out += s[j];
            result.flat(i) = out;
        }
    }
    return result;
}

template <class Container>
xarray_container<bool> str_startswith(const Container& strings, const std::string& pat) {
    xarray_container<bool> result(strings.shape());
    for (size_t i = 0; i < strings.size(); ++i) {
        const std::string& s = strings.flat(i);
        result.flat(i) = s.size() >= pat.size() && s.compare(0, pat.size(), pat) == 0;
    }
    return result;
}

template <class Container>
xarray_container<bool> str_endswith(const Container& strings, const std::string& pat) {
    xarray_container<bool> result(strings.shape());
    for (size_t i = 0; i < strings.size(); ++i) {
        const std::string& s = strings.flat(i);
        result.flat(i) = s.size() >= pat.size() && 
                         s.compare(s.size() - pat.size(), pat.size(), pat) == 0;
    }
    return result;
}

} // namespace string_ops

// Bring into frame namespace for convenience
using string_ops::StringAccessor;
using string_ops::str_accessor;
using string_ops::str_contains;
using string_ops::str_replace;
using string_ops::str_lower;
using string_ops::str_upper;
using string_ops::str_strip;
using string_ops::str_len;
using string_ops::str_slice;
using string_ops::str_startswith;
using string_ops::str_endswith;

} // namespace frame
XTU_NAMESPACE_END

#endif // XTU_FRAME_XSTRING_HPP