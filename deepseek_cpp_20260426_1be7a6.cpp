// File 142: modules/genesis/src/recorders/plotter.h
// Simple ASCII line plotter for 1D time‑series data recorded by BaseRecorder.
// Outputs a scaled graph to a string (for console or log file). Supports
// multiple channels, axis labels, and auto‑ranging.

#ifndef GENESIS_RECORDERS_PLOTTER_H
#define GENESIS_RECORDERS_PLOTTER_H

#include "base_recorder.h"
#include "core/string/ustring.h"
#include "core/templates/local_vector.h"
#include "core/typedefs.h"

namespace genesis {

class RecorderPlotter {
public:
	int  canvas_width = 80;           // characters horizontally
	int  canvas_height = 20;          // characters vertically
	bool show_grid = true;            // draw grid lines
	char point_char = '*';            // character for data points
	char axis_char = '+';

	RecorderPlotter() {}

	/**
	 * Generate an ASCII plot string for one or more channels in a recorder.
	 * @param recorder  Source recorder containing the data.
	 * @param channels  Names of channels to plot (must all have same length and numeric type).
	 * @param title     Optional title for the plot.
	 * @return          Multi‑line string with the plot.
	 */
	String plot(const Ref<BaseRecorder> &p_recorder,
				const LocalVector<String> &p_channels,
				const String &p_title = "Plot") const {
		ERR_FAIL_COND_V(p_recorder.is_null(), String());

		int n_channels = p_channels.size();
		if (n_channels == 0) return String();

		// Determine the number of frames (use the minimum across selected channels)
		int frames = p_recorder->get_current_frame();
		for (const String &ch : p_channels) {
			// Data access requires getting all data from the recorder.
			// get_all_data() returns a Dictionary of per‑channel Arrays.
		}
		Dictionary data_dict = p_recorder->get_all_data();
		LocalVector<LocalVector<real_t>> series(n_channels);
		int min_frames = frames;
		for (int c = 0; c < n_channels; ++c) {
			Array arr = data_dict.get(p_channels[c], Array());
			int n = arr.size();
			series[c].resize(n);
			for (int i = 0; i < n; ++i) {
				Variant val = arr[i];
				real_t v = (val.get_type() == Variant::FLOAT) ? (real_t)val : (val.get_type() == Variant::VECTOR3 ? ((Vector3)val).x : 0.0);
				series[c][i] = v;
			}
			min_frames = MIN(min_frames, n);
		}

		if (min_frames == 0) return String("No data.");

		// Determine global min and max across all series.
		real_t y_min = INFINITY, y_max = -INFINITY;
		for (int c = 0; c < n_channels; ++c) {
			for (int i = 0; i < min_frames; ++i) {
				real_t v = series[c][i];
				if (v < y_min) y_min = v;
				if (v > y_max) y_max = v;
			}
		}
		if (y_max - y_min < CMP_EPSILON) y_max = y_min + 1.0; // avoid zero range

		// 2D grid of characters (height rows, width columns).
		LocalVector<LocalVector<char>> screen(canvas_height);
		for (int r = 0; r < canvas_height; ++r) {
			screen[r].resize(canvas_width);
			for (int c = 0; c < canvas_width; ++c) screen[r][c] = ' ';
		}

		// Plot x-axis (frame index) along x direction, y-axis scaled.
		// Map frame index to column: col = (i / (min_frames-1)) * (canvas_width-1)
		// Map data value to row: row = (canvas_height-1) - ( (v - y_min)/(y_max - y_min) * (canvas_height-1) )
		auto val_to_row = [&](real_t v) -> int {
			real_t t = (v - y_min) / (y_max - y_min);
			return (canvas_height - 1) - (int)(t * (canvas_height - 1) + 0.5);
		};

		// Draw grid (if enabled) using dot chars
		if (show_grid) {
			for (int r = 0; r < canvas_height; ++r) {
				for (int c = 0; c < canvas_width; ++c) {
					if (r == 0 || r == canvas_height-1 || c == 0 || c == canvas_width-1 ||
						r % 5 == 0 || c % 10 == 0)
						screen[r][c] = '.';
				}
			}
		}

		// Draw x-axis at bottom row (canvas_height-1)
		for (int c = 0; c < canvas_width; ++c) screen[canvas_height-1][c] = '-';
		// Draw y-axis at first column
		for (int r = 0; r < canvas_height; ++r) screen[r][0] = '|';

		// Plot each series with a different marker (cycles through chars)
		const char *markers = "*#@OX"; // distinct markers
		for (int c = 0; c < n_channels; ++c) {
			char marker = markers[c % 5];
			Vector3 color; // not used in ASCII
			for (int i = 0; i < min_frames; ++i) {
				int col = (min_frames > 1) ? (int)((real_t)i / (min_frames-1) * (canvas_width-1) + 0.5) : 0;
				int row = val_to_row(series[c][i]);
				col = CLAMP(col, 0, canvas_width-1);
				row = CLAMP(row, 0, canvas_height-1);
				if (screen[row][col] == ' ' || screen[row][col] == '.')
					screen[row][col] = marker;
			}
		}

		// Assemble string with title and axis labels.
		String out;
		out += p_title + "\n";
		// Y-axis label at top
		out += vformat("Y max: %s\n", rtos(y_max));
		for (int r = 0; r < canvas_height; ++r) {
			for (int c = 0; c < canvas_width; ++c) {
				out += screen[r][c];
			}
			out += "\n";
		}
		out += vformat("Y min: %s   X: 0 .. %d frames\n", rtos(y_min), min_frames-1);
		return out;
	}

	/**
	 * Convenience method that prints the plot to Godot's output (print_line).
	 */
	void print_plot(const Ref<BaseRecorder> &p_recorder,
					const LocalVector<String> &p_channels,
					const String &p_title = "Plot") const {
		String s = plot(p_recorder, p_channels, p_title);
		print_line(s);
	}
};

} // namespace genesis

#endif // GENESIS_RECORDERS_PLOTTER_H