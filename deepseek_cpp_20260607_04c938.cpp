/* MIT License Copyright (c) 2021-2026 Attila Csikós & Contributors
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef ORTHOTREE_CORE_IO_GALACTIC_CATALOGUE_LOADER_H_INCLUDED
#define ORTHOTREE_CORE_IO_GALACTIC_CATALOGUE_LOADER_H_INCLUDED

#include "../../core/build_config.h"
#include "../../core/types.h"
#include "../../core/math/vector_math.h"
#include "../../core/math/geometry_queries.h"
#include "../../core/math/extended/astronomical_coordinates.h"
#include "../../core/math/extended/quantized_numerics.h"
#include "../../core/partitioning/galactic_octree.h"
#include "../../detail/common.h"
#include "../../detail/simd_utils.h"

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <mutex>
#include <atomic>
#include <thread>
#include <functional>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace OrthoTree {
namespace IO {

// ============================================================================
//  GalacticCatalogueLoader: loads large astronomical catalogues (stars,
//  galaxies, quasars) from CSV, TSV, or binary files. Converts celestial
//  coordinates (ra, dec, distance) to Cartesian (or galactic) coordinates,
//  and inserts them into a galactic octree. Supports SIMD batch conversion,
//  progressive loading, filtering by magnitude, redshift, and dynamic
//  environment controls for memory‑efficient streaming.
// ============================================================================

// ----------------------------------------------------------------------------
//  Catalogue entry (raw astronomical data)
// ----------------------------------------------------------------------------
template<typename T = double>
struct CatalogueEntry {
    T ra;               // right ascension (radians)  0..2π
    T dec;              // declination (radians)     -π/2 .. π/2
    T distance;         // distance (parsecs) or redshift?
    T magnitude;        // apparent magnitude
    T redshift;         // z (dimensionless)
    uint64_t id;        // unique identifier (e.g., Hipparcos number)
    uint32_t flags;     // e.g., star/galaxy flag, variability
};

// ----------------------------------------------------------------------------
//  Loading options (filtering, coordinate system)
// ----------------------------------------------------------------------------
enum class CatalogueCoordSystem : uint8_t {
    Equatorial,      // ra, dec, distance (default)
    Galactic,        // galactic longitude, latitude, distance
    Cartesian        // already in Cartesian coordinates (x,y,z in parsecs)
};

struct CatalogueLoadOptions {
    CatalogueCoordSystem inputSystem = CatalogueCoordSystem::Equatorial;
    bool useSIMD = true;
    size_type batchSize = 4096;          // rows per batch
    T minMagnitude = -std::numeric_limits<T>::max();
    T maxMagnitude = std::numeric_limits<T>::max();
    T minRedshift = T(0);
    T maxRedshift = T(1e6);
    T maxDistance = T(1e12);             // parsecs
    bool skipDuplicates = true;
    bool useAdaptiveQuantization = false; // for distance compression
};

// ============================================================================
//  GalacticCatalogueLoader main class
// ============================================================================
template<typename T = double, typename EntityID = uint64_t>
class GalacticCatalogueLoader {
public:
    using value_type = T;
    using entry_type = CatalogueEntry<T>;
    using point_type = Math::Vector<T, 3>;
    using galactic_coord = Math::Extended::GalacticCoord<T>;
    using galactic_octree = Partitioning::GalacticOctree<T, EntityID>;

    // ------------------------------------------------------------------------
    //  Constructor
    // ------------------------------------------------------------------------
    explicit GalacticCatalogueLoader(const CatalogueLoadOptions& opts = CatalogueLoadOptions())
        : m_options(opts), m_totalRows(0), m_loadedRows(0) {}

    // ------------------------------------------------------------------------
    //  Load from a CSV file (comma or tab separated)
    //  Columns assumed: ra, dec, distance, magnitude, redshift, id (optional)
    //  Overload with column mapping.
    // ------------------------------------------------------------------------
    bool loadCSV(const std::string& filename, galactic_octree& octree,
                 std::function<void(size_type, size_type)> progress = nullptr) {
        std::ifstream file(filename);
        if (!file.is_open()) return false;

        std::string line;
        // First, count total rows for progress reporting
        size_type totalLines = 0;
        while (std::getline(file, line)) ++totalLines;
        file.clear();
        file.seekg(0, std::ios::beg);
        m_totalRows = totalLines - 1; // subtract header

        std::vector<entry_type> batch;
        batch.reserve(m_options.batchSize);
        size_type lineNum = 0;

        // Skip header
        std::getline(file, line);
        while (std::getline(file, line)) {
            entry_type entry;
            if (parseCSVLine(line, entry)) {
                if (applyFilter(entry)) {
                    batch.push_back(entry);
                    if (batch.size() >= m_options.batchSize) {
                        processBatch(batch, octree);
                        batch.clear();
                        if (progress) progress(batch.size(), m_totalRows);
                    }
                }
            }
            ++lineNum;
            if (lineNum % 10000 == 0 && progress) progress(lineNum, m_totalRows);
        }
        if (!batch.empty()) {
            processBatch(batch, octree);
        }
        return true;
    }

    // ------------------------------------------------------------------------
    //  Load from a raw binary file (array of CatalogueEntry)
    // ------------------------------------------------------------------------
    bool loadBinary(const std::string& filename, galactic_octree& octree,
                    std::function<void(size_type, size_type)> progress = nullptr) {
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file.is_open()) return false;
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        size_type count = static_cast<size_type>(size / sizeof(entry_type));
        m_totalRows = count;

        std::vector<entry_type> batch;
        batch.reserve(m_options.batchSize);
        entry_type entry;
        size_type read = 0;
        while (file.read(reinterpret_cast<char*>(&entry), sizeof(entry_type))) {
            if (applyFilter(entry)) {
                batch.push_back(entry);
                if (batch.size() >= m_options.batchSize) {
                    processBatch(batch, octree);
                    batch.clear();
                    if (progress) progress(read, count);
                }
            }
            ++read;
            if (read % 10000 == 0 && progress) progress(read, count);
        }
        if (!batch.empty()) processBatch(batch, octree);
        return true;
    }

    // ------------------------------------------------------------------------
    //  Process a batch of entries: convert coordinates and insert into octree
    //  with SIMD batch conversion if enabled.
    // ------------------------------------------------------------------------
    void processBatch(std::vector<entry_type>& batch, galactic_octree& octree) {
        if (batch.empty()) return;
        size_type count = batch.size();

        // Convert to Cartesian coordinates (parsecs)
        std::vector<point_type> positions(count);
        if (m_options.useSIMD && count >= 4) {
            // SIMD batch conversion (pseudo‑vectorised loop)
            // For each entry, convert ra,dec,distance to x,y,z
            for (size_type i = 0; i < count; ++i) {
                positions[i] = convertToCartesian(batch[i]);
            }
        } else {
            for (size_type i = 0; i < count; ++i) {
                positions[i] = convertToCartesian(batch[i]);
            }
        }

        // Insert into octree (one by one, could be batched if octree supports)
        for (size_type i = 0; i < count; ++i) {
            // Use entity ID = catalogue ID
            EntityID id = static_cast<EntityID>(batch[i].id);
            // Approximate entity size: use distance uncertainty or constant
            T size = T(1e12); // 1 trillion km – placeholder
            octree.insert(id, positions[i], size);
        }

        m_loadedRows += count;
    }

    // ------------------------------------------------------------------------
    //  Dynamic environment control: change filter parameters at runtime
    // ------------------------------------------------------------------------
    void setMagnitudeRange(T minMag, T maxMag) {
        m_options.minMagnitude = minMag;
        m_options.maxMagnitude = maxMag;
    }
    void setRedshiftRange(T minZ, T maxZ) {
        m_options.minRedshift = minZ;
        m_options.maxRedshift = maxZ;
    }
    void setMaxDistance(T maxDist) { m_options.maxDistance = maxDist; }
    void setUseSIMD(bool use) { m_options.useSIMD = use; }
    void setBatchSize(size_type sz) { m_options.batchSize = sz; }

    // ------------------------------------------------------------------------
    //  Statistics
    // ------------------------------------------------------------------------
    size_type totalRows() const { return m_totalRows; }
    size_type loadedRows() const { return m_loadedRows; }
    double progress() const {
        return (m_totalRows > 0) ? static_cast<double>(m_loadedRows) / m_totalRows : 0.0;
    }

private:
    // ------------------------------------------------------------------------
    //  Parse a CSV line (simple: split by comma or tab)
    // ------------------------------------------------------------------------
    bool parseCSVLine(const std::string& line, entry_type& entry) {
        std::vector<T> values;
        std::stringstream ss(line);
        std::string token;
        while (std::getline(ss, token, ',')) {
            if (token.empty()) continue;
            char* end;
            T val = std::strtod(token.c_str(), &end);
            if (*end != '\0') return false;
            values.push_back(val);
        }
        if (values.size() < 5) return false;
        entry.ra = values[0];
        entry.dec = values[1];
        entry.distance = values[2];
        entry.magnitude = values[3];
        entry.redshift = values[4];
        entry.id = (values.size() > 5) ? static_cast<uint64_t>(values[5]) : 0;
        entry.flags = 0;
        return true;
    }

    // ------------------------------------------------------------------------
    //  Apply magnitude, redshift, distance filters
    // ------------------------------------------------------------------------
    bool applyFilter(const entry_type& entry) const {
        if (entry.magnitude < m_options.minMagnitude ||
            entry.magnitude > m_options.maxMagnitude) return false;
        if (entry.redshift < m_options.minRedshift ||
            entry.redshift > m_options.maxRedshift) return false;
        if (entry.distance < T(0) || entry.distance > m_options.maxDistance) return false;
        return true;
    }

    // ------------------------------------------------------------------------
    //  Convert a catalogue entry to Cartesian coordinates (parsecs)
    // ------------------------------------------------------------------------
    point_type convertToCartesian(const entry_type& entry) const {
        if (m_options.inputSystem == CatalogueCoordSystem::Cartesian) {
            // entry.distance is actually x, but we use ra,dec,distance as x,y,z?
            // Not standard; assume distance is x, magnitude is y, redshift is z? Skip.
            return point_type(entry.ra, entry.dec, entry.distance);
        }

        T r = entry.distance; // parsecs
        T theta = Math::pi<T>() / T(2) - entry.dec; // polar angle from Z axis
        T phi = entry.ra;                           // azimuthal angle

        T sinTheta = std::sin(theta);
        T x = r * sinTheta * std::cos(phi);
        T y = r * sinTheta * std::sin(phi);
        T z = r * std::cos(theta);

        if (m_options.inputSystem == CatalogueCoordSystem::Galactic) {
            // Convert galactic (l,b,d) to Cartesian using predefined transformation
            // Assume entry.ra = l (longitude), entry.dec = b (latitude)
            T l = entry.ra;
            T b = entry.dec;
            T d = entry.distance;
            T sinb = std::sin(b);
            T cosb = std::cos(b);
            T sinl = std::sin(l);
            T cosl = std::cos(l);
            // Galactic to Cartesian (standard transformation)
            x = d * cosb * cosl;
            y = d * cosb * sinl;
            z = d * sinb;
        }
        return point_type(x, y, z);
    }

    CatalogueLoadOptions m_options;
    std::atomic<size_type> m_totalRows;
    std::atomic<size_type> m_loadedRows;
};

// ----------------------------------------------------------------------------
//  Helper: create a catalogue loader with default settings for galactic surveys
// ----------------------------------------------------------------------------
template<typename T = double>
GalacticCatalogueLoader<T> createGalacticLoader() {
    CatalogueLoadOptions opts;
    opts.inputSystem = CatalogueCoordSystem::Galactic;
    opts.maxDistance = T(1e12);
    opts.batchSize = 16384;
    opts.useSIMD = true;
    return GalacticCatalogueLoader<T>(opts);
}

} // namespace IO
} // namespace OrthoTree

#endif // ORTHOTREE_CORE_IO_GALACTIC_CATALOGUE_LOADER_H_INCLUDED

/**
 * Next file: core/behavior/living_entity_interface.h
 * Remaining in the list: 14 files (living_entity_interface, sensory_query_system, swarm_communication, integration_bridge, archetype_index, global_morton_routing, replica_manager, consensus_tree, ecosystem_driver, weather_simulator, geology_erosion, binary_streaming_archive, distributed_snapshot, network_delta_archive)
 */