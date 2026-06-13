// genesis/_main.h

#pragma once

#include <string>
#include <vector>
#include <functional>
#include <map>

namespace genesis {
namespace main_entry {

// Command-line argument structure
struct Args {
    bool help = false;
    bool version = false;
    std::string script_path;
    std::vector<std::string> script_args;
    std::string backend = "gpu";
    std::string precision = "32";
    bool debug = false;
    bool performance_mode = false;
    uint64_t seed = 0;
    std::string logging_level = "info";
    std::string theme = "dark";
};

// Parse command line arguments
Args parse_args(int argc, char* argv[]);

// Print usage information
void print_usage(const std::string& program_name);

// Print version banner
void print_version();

// Main entry point function (called from actual main)
int main_entry(int argc, char* argv[]);

// Register a custom script runner (for embedding)
using ScriptRunner = std::function<int(const std::string& path, const std::vector<std::string>& args)>;
void register_script_runner(ScriptRunner runner);

} // namespace main_entry
} // namespace genesis