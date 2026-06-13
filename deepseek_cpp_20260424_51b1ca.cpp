// genesis/_main.cpp

#include "genesis/_main.h"
#include "genesis/__init__.h"
#include "genesis/version.h"
#include "genesis/constants.h"
#include "genesis/engine/simulator.h"
#include "genesis/engine/scene.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <termios.h>
#include <sys/ioctl.h>
#endif

namespace genesis {
namespace main_entry {

namespace {
    // Internal script runner (default: execute Python-like script via embedded interpreter)
    ScriptRunner g_script_runner = nullptr;
    
    // Helper to check if file exists
    bool file_exists(const std::string& path) {
        std::ifstream f(path);
        return f.good();
    }
    
    // Helper to get terminal width
    int get_terminal_width() {
#ifdef _WIN32
        CONSOLE_SCREEN_BUFFER_INFO csbi;
        if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi)) {
            return csbi.srWindow.Right - csbi.srWindow.Left + 1;
        }
#else
        struct winsize w;
        if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0) {
            return w.ws_col;
        }
#endif
        return 80; // default
    }
    
    // Print centered text with borders
    void print_banner_line(const std::string& text, char border = '=') {
        int width = get_terminal_width();
        if (text.empty()) {
            std::cout << std::string(width, border) << std::endl;
        } else {
            int padding = (width - static_cast<int>(text.length()) - 2) / 2;
            if (padding < 0) padding = 0;
            std::cout << std::string(padding, ' ') << " " << text << " "
                      << std::string(width - padding - static_cast<int>(text.length()) - 2, ' ')
                      << std::endl;
        }
    }
    
    // Convert string to lowercase
    std::string to_lower(std::string s) {
        std::transform(s.begin(), s.end(), s.begin(), ::tolower);
        return s;
    }
    
    // Map logging level string to enum
    Logger::Level parse_logging_level(const std::string& level) {
        std::string l = to_lower(level);
        if (l == "debug") return Logger::Level::DEBUG;
        if (l == "info") return Logger::Level::INFO;
        if (l == "warning" || l == "warn") return Logger::Level::WARNING;
        if (l == "error") return Logger::Level::ERROR;
        return Logger::Level::INFO;
    }
    
    // Map backend string to enum
    Backend parse_backend(const std::string& backend) {
        std::string b = to_lower(backend);
        if (b == "cpu") return Backend::CPU;
        if (b == "gpu") return Backend::GPU;
        if (b == "cuda") return Backend::CUDA;
        if (b == "amdgpu" || b == "amd") return Backend::AMDGPU;
        if (b == "metal") return Backend::METAL;
        return Backend::GPU;
    }
} // anonymous namespace

//------------------------------------------------------------------------------
// parse_args
//------------------------------------------------------------------------------
Args parse_args(int argc, char* argv[]) {
    Args args;
    std::vector<std::string> positional;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            args.help = true;
        } else if (arg == "-v" || arg == "--version") {
            args.version = true;
        } else if (arg == "--debug") {
            args.debug = true;
        } else if (arg == "--performance") {
            args.performance_mode = true;
        } else if (arg.find("--backend=") == 0) {
            args.backend = arg.substr(10);
        } else if (arg.find("--precision=") == 0) {
            args.precision = arg.substr(12);
        } else if (arg.find("--seed=") == 0) {
            args.seed = std::stoull(arg.substr(7));
        } else if (arg.find("--log-level=") == 0) {
            args.logging_level = arg.substr(12);
        } else if (arg.find("--theme=") == 0) {
            args.theme = arg.substr(8);
        } else if (arg == "--") {
            // Everything after -- goes to script args
            for (int j = i + 1; j < argc; ++j) {
                args.script_args.push_back(argv[j]);
            }
            break;
        } else if (arg[0] != '-') {
            if (args.script_path.empty()) {
                args.script_path = arg;
            } else {
                args.script_args.push_back(arg);
            }
        } else {
            std::cerr << "Warning: Unknown option: " << arg << std::endl;
        }
    }
    return args;
}

//------------------------------------------------------------------------------
// print_usage
//------------------------------------------------------------------------------
void print_usage(const std::string& program_name) {
    std::cout << "Genesis Physics Engine v" << VERSION << "\n\n";
    std::cout << "Usage: " << program_name << " [options] [script] [script_args...]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -h, --help            Show this help message and exit\n";
    std::cout << "  -v, --version         Show version information and exit\n";
    std::cout << "  --backend=BACKEND     Set compute backend (cpu, gpu, cuda, amdgpu, metal) [default: gpu]\n";
    std::cout << "  --precision=PREC      Set floating point precision (32 or 64) [default: 32]\n";
    std::cout << "  --debug               Enable debug mode (forces CPU backend, verbose logging)\n";
    std::cout << "  --performance         Enable performance mode (disables safety checks)\n";
    std::cout << "  --seed=N              Set random seed for reproducibility\n";
    std::cout << "  --log-level=LEVEL     Set logging level (debug, info, warning, error) [default: info]\n";
    std::cout << "  --theme=THEME         Set visualization theme (dark, light, dumb) [default: dark]\n";
    std::cout << "  --                    Stop parsing options; everything after goes to script\n";
    std::cout << "\nIf no script is provided, Genesis starts in interactive mode.\n";
}

//------------------------------------------------------------------------------
// print_version
//------------------------------------------------------------------------------
void print_version() {
    int width = get_terminal_width();
    std::string border(width, '=');
    std::cout << border << std::endl;
    print_banner_line("GENESIS", '=');
    print_banner_line("Generalized Embodied Intelligence Simulation Platform", ' ');
    std::cout << border << std::endl;
    std::cout << "Version: " << VERSION << "\n";
    std::cout << "Build: " << __DATE__ << " " << __TIME__ << "\n";
    std::cout << "Backend support: CPU, CUDA, AMDGPU, Metal\n";
    std::cout << border << std::endl;
}

//------------------------------------------------------------------------------
// register_script_runner
//------------------------------------------------------------------------------
void register_script_runner(ScriptRunner runner) {
    g_script_runner = std::move(runner);
}

//------------------------------------------------------------------------------
// main_entry
//------------------------------------------------------------------------------
int main_entry(int argc, char* argv[]) {
    Args args = parse_args(argc, argv);
    
    // Handle help and version requests immediately
    if (args.help) {
        print_usage(argv[0]);
        return 0;
    }
    
    if (args.version) {
        print_version();
        return 0;
    }
    
    // Print startup banner (unless in dumb theme)
    if (args.theme != "dumb") {
        int width = get_terminal_width();
        std::cout << std::string(width, '=') << std::endl;
        std::cout << "Genesis Physics Engine v" << VERSION << std::endl;
        std::cout << std::string(width, '=') << std::endl;
    }
    
    // Initialize Genesis
    try {
        Backend backend_enum = parse_backend(args.backend);
        Logger::Level log_level = parse_logging_level(args.logging_level);
        
        init(backend_enum, args.precision, log_level, args.debug,
             args.seed, 1e-15, args.theme, false, args.performance_mode);
    } catch (const GenesisException& e) {
        std::cerr << "Initialization failed: " << e.what() << std::endl;
        return 1;
    }
    
    // Run script if provided
    if (!args.script_path.empty()) {
        if (!file_exists(args.script_path)) {
            std::cerr << "Error: Script file not found: " << args.script_path << std::endl;
            destroy();
            return 1;
        }
        
        // Execute script using registered runner (default: internal Python interpreter)
        int ret = 0;
        if (g_script_runner) {
            ret = g_script_runner(args.script_path, args.script_args);
        } else {
            // Fallback: if no runner registered, we can try to execute as a Genesis scene description file
            // For demonstration, we'll attempt to load it as a simulation configuration
            std::cerr << "Warning: No script runner registered. Attempting to load as scene description...\n";
            try {
                auto scene = std::make_shared<Scene>();
                // In actual implementation, scene would parse the file and build simulation
                // This is a placeholder for the actual file parsing logic.
                std::cerr << "Error: Scene loading from file not yet implemented in this standalone build.\n";
                ret = 1;
            } catch (const std::exception& e) {
                std::cerr << "Error loading scene: " << e.what() << std::endl;
                ret = 1;
            }
        }
        
        destroy();
        return ret;
    } else {
        // Interactive mode (if supported)
        std::cout << "Genesis interactive mode (type 'exit' to quit)\n";
        std::cout << ">>> " << std::flush;
        
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line == "exit" || line == "quit") break;
            if (line.empty()) {
                std::cout << ">>> " << std::flush;
                continue;
            }
            // Execute command in interactive context
            std::cout << "Executed: " << line << std::endl;
            std::cout << ">>> " << std::flush;
        }
        
        destroy();
        return 0;
    }
}

} // namespace main_entry
} // namespace genesis

//------------------------------------------------------------------------------
// Standard main function (entry point for executable)
//------------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    return genesis::main_entry::main_entry(argc, argv);
}