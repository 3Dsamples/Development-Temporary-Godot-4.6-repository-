// image/xtexture_compressor.hpp

#ifndef XTENSOR_XTEXTURE_COMPRESSOR_HPP
#define XTENSOR_XTEXTURE_COMPRESSOR_HPP

#include "../core/xtensor_config.hpp"
#include "../core/xtensor_forward.hpp"
#include "../core/xexpression.hpp"
#include "../containers/xarray.hpp"
#include "../containers/xtensor.hpp"
#include "../core/xview.hpp"
#include "../math/xstats.hpp"
#include "../math/xnorm.hpp"
#include "../math/xlinalg.hpp"
#include "ximage_processing.hpp"

#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <functional>
#include <stdexcept>
#include <limits>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <complex>
#include <map>
#include <queue>
#include <tuple>
#include <unordered_map>

namespace xt
{
    XTENSOR_INLINE_NAMESPACE
    {
        namespace texture
        {
            using Image = xarray_container<double>;
            using ImageU8 = xarray_container<uint8_t>;
            using ImageRGBA = xarray_container<uint8_t>; // HxWx4

            // --------------------------------------------------------------------
            // Block Compression Formats (BCn / DXT)
            // --------------------------------------------------------------------
            namespace bc
            {
                // DXT1 (BC1) - 4x4 blocks, 64 bits per block (RGB, 1-bit alpha)
                struct DXT1Block
                {
                    uint16_t color0;      // 565 RGB
                    uint16_t color1;      // 565 RGB
                    uint32_t indices;     // 2 bits per pixel (16 pixels)
                };

                // DXT5 (BC3) - 4x4 blocks, 128 bits per block (RGBA)
                struct DXT5Block
                {
                    uint8_t alpha0;
                    uint8_t alpha1;
                    uint8_t alpha_indices[6]; // 48 bits = 3 bits per pixel * 16
                    uint16_t color0;           // 565 RGB
                    uint16_t color1;           // 565 RGB
                    uint32_t color_indices;    // 2 bits per pixel * 16
                };

                // Helper: convert 8-bit RGB to 565
                inline uint16_t rgb_to_565(uint8_t r, uint8_t g, uint8_t b)
                {
                    return static_cast<uint16_t>(((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3));
                }

                // Helper: convert 565 to 8-bit RGB (array of 3 uint8_t)
                inline std::array<uint8_t, 3> rgb565_to_888(uint16_t c)
                {
                    uint8_t r = static_cast<uint8_t>((c >> 11) & 0x1F) << 3;
                    uint8_t g = static_cast<uint8_t>((c >> 5) & 0x3F) << 2;
                    uint8_t b = static_cast<uint8_t>(c & 0x1F) << 3;
                    // Replicate high bits into low bits
                    r |= r >> 5;
                    g |= g >> 6;
                    b |= b >> 5;
                    return {r, g, b};
                }

                // Compute palette for BC1: 4 colors (c0, c1, c2, c3)
                inline std::array<std::array<uint8_t, 3>, 4> bc1_palette(uint16_t c0, uint16_t c1)
                {
                    auto col0 = rgb565_to_888(c0);
                    auto col1 = rgb565_to_888(c1);
                    std::array<std::array<uint8_t, 3>, 4> pal;
                    pal[0] = col0;
                    pal[1] = col1;
                    if (c0 > c1)
                    {
                        // c2 = 2/3*c0 + 1/3*c1, c3 = 1/3*c0 + 2/3*c1
                        for (int i = 0; i < 3; ++i)
                        {
                            pal[2][i] = static_cast<uint8_t>((2 * col0[i] + 1 * col1[i]) / 3);
                            pal[3][i] = static_cast<uint8_t>((1 * col0[i] + 2 * col1[i]) / 3);
                        }
                    }
                    else
                    {
                        // c2 = 1/2*(c0+c1), c3 = transparent black
                        for (int i = 0; i < 3; ++i)
                            pal[2][i] = static_cast<uint8_t>((col0[i] + col1[i]) / 2);
                        pal[3] = {0, 0, 0};
                    }
                    return pal;
                }

                // Compute alpha palette for BC3 (8 alpha values from a0, a1)
                inline std::array<uint8_t, 8> bc3_alpha_palette(uint8_t a0, uint8_t a1)
                {
                    std::array<uint8_t, 8> pal;
                    pal[0] = a0;
                    pal[1] = a1;
                    if (a0 > a1)
                    {
                        // 6 interpolated values
                        for (int i = 1; i <= 6; ++i)
                            pal[i+1] = static_cast<uint8_t>(((7 - i) * a0 + i * a1) / 7);
                    }
                    else
                    {
                        // 4 interpolated, then 0 and 255
                        for (int i = 1; i <= 4; ++i)
                            pal[i+1] = static_cast<uint8_t>(((5 - i) * a0 + i * a1) / 5);
                        pal[6] = 0;
                        pal[7] = 255;
                    }
                    return pal;
                }

                // Encode a 4x4 RGB block to BC1
                inline DXT1Block compress_bc1_block(const ImageU8& block) // block: 4x4x3
                {
                    if (block.shape()[0] != 4 || block.shape()[1] != 4)
                        XTENSOR_THROW(std::invalid_argument, "compress_bc1_block: expected 4x4 block");

                    // Find min and max colors in RGB space (using simple bounding box)
                    uint8_t min_r = 255, min_g = 255, min_b = 255;
                    uint8_t max_r = 0, max_g = 0, max_b = 0;
                    for (size_t y = 0; y < 4; ++y)
                    {
                        for (size_t x = 0; x < 4; ++x)
                        {
                            uint8_t r = block(y, x, 0);
                            uint8_t g = block(y, x, 1);
                            uint8_t b = block(y, x, 2);
                            min_r = std::min(min_r, r);
                            min_g = std::min(min_g, g);
                            min_b = std::min(min_b, b);
                            max_r = std::max(max_r, r);
                            max_g = std::max(max_g, g);
                            max_b = std::max(max_b, b);
                        }
                    }

                    // Use principal component analysis to find better endpoints? For simplicity, use min/max.
                    uint16_t c0 = rgb_to_565(max_r, max_g, max_b);
                    uint16_t c1 = rgb_to_565(min_r, min_g, min_b);
                    if (c0 < c1) std::swap(c0, c1); // ensure c0 > c1 for opaque mode

                    auto palette = bc1_palette(c0, c1);
                    DXT1Block blk;
                    blk.color0 = c0;
                    blk.color1 = c1;
                    blk.indices = 0;

                    for (size_t y = 0; y < 4; ++y)
                    {
                        for (size_t x = 0; x < 4; ++x)
                        {
                            uint8_t r = block(y, x, 0);
                            uint8_t g = block(y, x, 1);
                            uint8_t b = block(y, x, 2);
                            // Find closest palette entry
                            int best_idx = 0;
                            double best_dist = std::numeric_limits<double>::max();
                            for (int i = 0; i < 4; ++i)
                            {
                                int dr = r - palette[i][0];
                                int dg = g - palette[i][1];
                                int db = b - palette[i][2];
                                double dist = dr*dr + dg*dg + db*db;
                                if (dist < best_dist)
                                {
                                    best_dist = dist;
                                    best_idx = i;
                                }
                            }
                            // Store 2 bits per pixel
                            size_t pixel_idx = y * 4 + x;
                            blk.indices |= static_cast<uint32_t>(best_idx) << (2 * pixel_idx);
                        }
                    }
                    return blk;
                }

                // Compress full RGBA image to BC3 (DXT5)
                inline std::vector<DXT5Block> compress_bc3(const ImageU8& img) // HxWx4
                {
                    if (img.dimension() != 3 || img.shape()[2] != 4)
                        XTENSOR_THROW(std::invalid_argument, "compress_bc3: expected HxWx4 image");
                    size_t h = img.shape()[0];
                    size_t w = img.shape()[1];
                    size_t blocks_h = (h + 3) / 4;
                    size_t blocks_w = (w + 3) / 4;
                    std::vector<DXT5Block> blocks(blocks_h * blocks_w);

                    for (size_t by = 0; by < blocks_h; ++by)
                    {
                        for (size_t bx = 0; bx < blocks_w; ++bx)
                        {
                            // Extract 4x4 block (pad with edge replication)
                            ImageU8 block = xt::zeros<uint8_t>({4, 4, 4});
                            for (size_t y = 0; y < 4; ++y)
                            {
                                for (size_t x = 0; x < 4; ++x)
                                {
                                    size_t sy = std::min(by * 4 + y, h - 1);
                                    size_t sx = std::min(bx * 4 + x, w - 1);
                                    for (size_t c = 0; c < 4; ++c)
                                        block(y, x, c) = img(sy, sx, c);
                                }
                            }

                            DXT5Block blk;
                            // Alpha compression
                            uint8_t min_a = 255, max_a = 0;
                            for (size_t y = 0; y < 4; ++y)
                                for (size_t x = 0; x < 4; ++x)
                                {
                                    uint8_t a = block(y, x, 3);
                                    min_a = std::min(min_a, a);
                                    max_a = std::max(max_a, a);
                                }
                            blk.alpha0 = max_a;
                            blk.alpha1 = min_a;
                            auto alpha_pal = bc3_alpha_palette(blk.alpha0, blk.alpha1);

                            // Fill alpha indices (3 bits each)
                            std::memset(blk.alpha_indices, 0, sizeof(blk.alpha_indices));
                            for (size_t y = 0; y < 4; ++y)
                            {
                                for (size_t x = 0; x < 4; ++x)
                                {
                                    uint8_t a = block(y, x, 3);
                                    int best = 0;
                                    int best_diff = 256;
                                    for (int i = 0; i < 8; ++i)
                                    {
                                        int diff = std::abs(a - alpha_pal[i]);
                                        if (diff < best_diff)
                                        {
                                            best_diff = diff;
                                            best = i;
                                        }
                                    }
                                    size_t pixel_idx = y * 4 + x;
                                    size_t byte_idx = pixel_idx * 3 / 8;
                                    size_t bit_offset = (pixel_idx * 3) % 8;
                                    // Store 3 bits (simplified: not handling bit boundaries correctly for brevity, but conceptually correct)
                                    // In a real implementation, we'd pack carefully.
                                    blk.alpha_indices[byte_idx] |= static_cast<uint8_t>(best << bit_offset);
                                    if (bit_offset > 5) // crosses byte boundary
                                    {
                                        blk.alpha_indices[byte_idx + 1] |= static_cast<uint8_t>(best >> (8 - bit_offset));
                                    }
                                }
                            }

                            // Color compression (BC1 part)
                            uint8_t min_r=255, min_g=255, min_b=255, max_r=0, max_g=0, max_b=0;
                            for (size_t y=0; y<4; ++y)
                                for (size_t x=0; x<4; ++x)
                                {
                                    uint8_t r = block(y,x,0), g = block(y,x,1), b = block(y,x,2);
                                    min_r = std::min(min_r, r); max_r = std::max(max_r, r);
                                    min_g = std::min(min_g, g); max_g = std::max(max_g, g);
                                    min_b = std::min(min_b, b); max_b = std::max(max_b, b);
                                }
                            uint16_t c0 = rgb_to_565(max_r, max_g, max_b);
                            uint16_t c1 = rgb_to_565(min_r, min_g, min_b);
                            if (c0 < c1) std::swap(c0, c1);
                            blk.color0 = c0;
                            blk.color1 = c1;
                            auto col_pal = bc1_palette(c0, c1);
                            blk.color_indices = 0;
                            for (size_t y=0; y<4; ++y)
                                for (size_t x=0; x<4; ++x)
                                {
                                    uint8_t r = block(y,x,0), g = block(y,x,1), b = block(y,x,2);
                                    int best = 0;
                                    double best_dist = 1e30;
                                    for (int i=0; i<4; ++i)
                                    {
                                        int dr = r - col_pal[i][0];
                                        int dg = g - col_pal[i][1];
                                        int db = b - col_pal[i][2];
                                        double d = dr*dr + dg*dg + db*db;
                                        if (d < best_dist) { best_dist = d; best = i; }
                                    }
                                    size_t idx = y*4 + x;
                                    blk.color_indices |= static_cast<uint32_t>(best) << (2*idx);
                                }

                            blocks[by * blocks_w + bx] = blk;
                        }
                    }
                    return blocks;
                }

                // Decompress BC3 to RGBA image
                inline ImageU8 decompress_bc3(const std::vector<DXT5Block>& blocks, size_t width, size_t height)
                {
                    size_t blocks_w = (width + 3) / 4;
                    size_t blocks_h = (height + 3) / 4;
                    if (blocks.size() != blocks_w * blocks_h)
                        XTENSOR_THROW(std::invalid_argument, "decompress_bc3: block count mismatch");

                    ImageU8 img = xt::zeros<uint8_t>({height, width, 4});
                    for (size_t by = 0; by < blocks_h; ++by)
                    {
                        for (size_t bx = 0; bx < blocks_w; ++bx)
                        {
                            const auto& blk = blocks[by * blocks_w + bx];
                            auto alpha_pal = bc3_alpha_palette(blk.alpha0, blk.alpha1);
                            auto col_pal = bc1_palette(blk.color0, blk.color1);

                            for (size_t y = 0; y < 4; ++y)
                            {
                                for (size_t x = 0; x < 4; ++x)
                                {
                                    size_t px = bx * 4 + x;
                                    size_t py = by * 4 + y;
                                    if (px >= width || py >= height) continue;
                                    size_t pixel_idx = y * 4 + x;
                                    // Decode alpha
                                    size_t alpha_byte = pixel_idx * 3 / 8;
                                    size_t alpha_bit = (pixel_idx * 3) % 8;
                                    uint8_t alpha_idx = (blk.alpha_indices[alpha_byte] >> alpha_bit) & 0x7;
                                    if (alpha_bit > 5)
                                    {
                                        uint8_t next = blk.alpha_indices[alpha_byte + 1];
                                        alpha_idx |= (next << (8 - alpha_bit)) & 0x7;
                                    }
                                    img(py, px, 3) = alpha_pal[alpha_idx];

                                    // Decode color
                                    uint8_t col_idx = (blk.color_indices >> (2 * pixel_idx)) & 0x3;
                                    auto col = col_pal[col_idx];
                                    img(py, px, 0) = col[0];
                                    img(py, px, 1) = col[1];
                                    img(py, px, 2) = col[2];
                                }
                            }
                        }
                    }
                    return img;
                }
            } // namespace bc

            // --------------------------------------------------------------------
            // ETC / ETC2 Compression (simplified)
            // --------------------------------------------------------------------
            namespace etc
            {
                struct ETC1Block
                {
                    uint64_t data;
                };

                // Placeholder: basic ETC1 compression using differential mode
                inline ETC1Block compress_etc1_block(const ImageU8& block)
                {
                    ETC1Block blk;
                    blk.data = 0;
                    // In a full implementation, this would encode using ETC1 algorithm
                    return blk;
                }
            }

            // --------------------------------------------------------------------
            // ASTC Compression (Adaptive Scalable Texture Compression)
            // --------------------------------------------------------------------
            namespace astc
            {
                // ASTC block sizes: 4x4 to 12x12
                struct ASTCBlock
                {
                    std::array<uint8_t, 16> data;
                };

                inline ASTCBlock compress_astc_block(const ImageU8& block, size_t block_w, size_t block_h)
                {
                    ASTCBlock blk;
                    std::memset(blk.data.data(), 0, 16);
                    // ASTC is complex; placeholder for interface
                    return blk;
                }
            }

            // --------------------------------------------------------------------
            // Mipmap generation
            // --------------------------------------------------------------------
            inline std::vector<ImageU8> generate_mipmaps(const ImageU8& img, size_t levels = 0)
            {
                size_t w = img.shape()[1];
                size_t h = img.shape()[0];
                if (levels == 0)
                    levels = static_cast<size_t>(std::floor(std::log2(std::max(w, h)))) + 1;
                std::vector<ImageU8> mips;
                mips.push_back(img);
                ImageU8 current = img;
                for (size_t lvl = 1; lvl < levels; ++lvl)
                {
                    size_t new_w = std::max(1UL, current.shape()[1] / 2);
                    size_t new_h = std::max(1UL, current.shape()[0] / 2);
                    ImageU8 down({new_h, new_w, current.shape()[2]});
                    for (size_t y = 0; y < new_h; ++y)
                    {
                        for (size_t x = 0; x < new_w; ++x)
                        {
                            for (size_t c = 0; c < current.shape()[2]; ++c)
                            {
                                // 2x2 box filter
                                size_t sy0 = y * 2, sy1 = std::min(sy0 + 1, current.shape()[0] - 1);
                                size_t sx0 = x * 2, sx1 = std::min(sx0 + 1, current.shape()[1] - 1);
                                uint16_t sum = static_cast<uint16_t>(current(sy0, sx0, c)) +
                                               static_cast<uint16_t>(current(sy0, sx1, c)) +
                                               static_cast<uint16_t>(current(sy1, sx0, c)) +
                                               static_cast<uint16_t>(current(sy1, sx1, c));
                                down(y, x, c) = static_cast<uint8_t>((sum + 2) / 4);
                            }
                        }
                    }
                    mips.push_back(down);
                    current = down;
                }
                return mips;
            }

            // --------------------------------------------------------------------
            // Normal map generation from heightmap
            // --------------------------------------------------------------------
            inline ImageU8 height_to_normal(const ImageU8& heightmap, double strength = 1.0)
            {
                if (heightmap.dimension() != 2)
                    XTENSOR_THROW(std::invalid_argument, "height_to_normal: expected 2D heightmap");
                size_t h = heightmap.shape()[0];
                size_t w = heightmap.shape()[1];
                ImageU8 normal({h, w, 3});
                for (size_t y = 0; y < h; ++y)
                {
                    for (size_t x = 0; x < w; ++x)
                    {
                        // Sobel operator to get gradients
                        int gx = 0, gy = 0;
                        if (x > 0 && x < w-1)
                        {
                            gx = -static_cast<int>(heightmap(y, x-1)) + static_cast<int>(heightmap(y, x+1));
                        }
                        if (y > 0 && y < h-1)
                        {
                            gy = -static_cast<int>(heightmap(y-1, x)) + static_cast<int>(heightmap(y+1, x));
                        }
                        double nx = -gx * strength;
                        double ny = -gy * strength;
                        double nz = 1.0;
                        double len = std::sqrt(nx*nx + ny*ny + nz*nz);
                        if (len > 0)
                        {
                            nx /= len;
                            ny /= len;
                            nz /= len;
                        }
                        // Convert to [0,255]
                        normal(y, x, 0) = static_cast<uint8_t>((nx * 0.5 + 0.5) * 255);
                        normal(y, x, 1) = static_cast<uint8_t>((ny * 0.5 + 0.5) * 255);
                        normal(y, x, 2) = static_cast<uint8_t>((nz * 0.5 + 0.5) * 255);
                    }
                }
                return normal;
            }

            // --------------------------------------------------------------------
            // DDS file format writer/reader (simplified)
            // --------------------------------------------------------------------
            struct DDSHeader
            {
                uint32_t magic = 0x20534444; // "DDS "
                uint32_t size = 124;
                uint32_t flags;
                uint32_t height;
                uint32_t width;
                uint32_t pitchOrLinearSize;
                uint32_t depth;
                uint32_t mipMapCount;
                uint32_t reserved1[11];
                struct {
                    uint32_t size;
                    uint32_t flags;
                    uint32_t fourCC;
                    uint32_t rgbBitCount;
                    uint32_t rBitMask, gBitMask, bBitMask, aBitMask;
                } pixelFormat;
                uint32_t caps;
                uint32_t caps2, caps3, caps4;
                uint32_t reserved2;
            };

            inline void write_dds(const std::string& filename, const std::vector<bc::DXT5Block>& blocks,
                                  size_t width, size_t height, size_t mip_levels = 1)
            {
                DDSHeader hdr = {};
                hdr.flags = 0x000A1007; // CAPS, HEIGHT, WIDTH, PIXELFORMAT, MIPMAPCOUNT, LINEARSIZE
                hdr.height = static_cast<uint32_t>(height);
                hdr.width = static_cast<uint32_t>(width);
                hdr.pitchOrLinearSize = static_cast<uint32_t>(blocks.size() * 16);
                hdr.mipMapCount = static_cast<uint32_t>(mip_levels);
                hdr.pixelFormat.size = 32;
                hdr.pixelFormat.flags = 0x4; // FOURCC
                hdr.pixelFormat.fourCC = 0x35545844; // "DXT5"
                hdr.caps = 0x401008; // TEXTURE, MIPMAP, COMPLEX

                std::ofstream file(filename, std::ios::binary);
                file.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
                file.write(reinterpret_cast<const char*>(blocks.data()), blocks.size() * sizeof(bc::DXT5Block));
                // Mipmaps would follow...
            }

            // --------------------------------------------------------------------
            // Texture atlas packing (bin packing)
            // --------------------------------------------------------------------
            struct Rect
            {
                size_t x, y, w, h;
            };

            inline std::vector<Rect> pack_rectangles(std::vector<std::pair<size_t, size_t>>& sizes, size_t atlas_w, size_t atlas_h)
            {
                // Simple shelf packing algorithm
                std::sort(sizes.begin(), sizes.end(), [](const auto& a, const auto& b) {
                    return a.second > b.second; // sort by height
                });
                std::vector<Rect> result(sizes.size());
                size_t shelf_y = 0, shelf_h = 0, shelf_x = 0;
                for (size_t i = 0; i < sizes.size(); ++i)
                {
                    size_t w = sizes[i].first;
                    size_t h = sizes[i].second;
                    if (w > atlas_w) XTENSOR_THROW(std::runtime_error, "Image width exceeds atlas width");
                    if (h > atlas_h) XTENSOR_THROW(std::runtime_error, "Image height exceeds atlas height");
                    if (shelf_x + w > atlas_w)
                    {
                        shelf_y += shelf_h;
                        shelf_x = 0;
                        shelf_h = 0;
                    }
                    if (shelf_y + h > atlas_h)
                        XTENSOR_THROW(std::runtime_error, "Atlas out of space");
                    result[i] = {shelf_x, shelf_y, w, h};
                    shelf_x += w;
                    shelf_h = std::max(shelf_h, h);
                }
                return result;
            }

            // Build atlas from images
            inline ImageU8 build_texture_atlas(const std::vector<ImageU8>& images,
                                                const std::vector<Rect>& rects,
                                                size_t atlas_w, size_t atlas_h)
            {
                size_t channels = images[0].shape()[2];
                ImageU8 atlas = xt::zeros<uint8_t>({atlas_h, atlas_w, channels});
                for (size_t i = 0; i < images.size(); ++i)
                {
                    const auto& img = images[i];
                    const auto& r = rects[i];
                    for (size_t y = 0; y < r.h; ++y)
                    {
                        for (size_t x = 0; x < r.w; ++x)
                        {
                            for (size_t c = 0; c < channels; ++c)
                            {
                                atlas(r.y + y, r.x + x, c) = img(y, x, c);
                            }
                        }
                    }
                }
                return atlas;
            }

            // --------------------------------------------------------------------
            // KTX2 file format (basic support)
            // --------------------------------------------------------------------
            inline void write_ktx2(const std::string& filename, const std::vector<uint8_t>& compressed_data,
                                   size_t width, size_t height, uint32_t format)
            {
                // Simplified KTX2 writer
                std::ofstream file(filename, std::ios::binary);
                // KTX2 identifier
                const uint8_t identifier[12] = {0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A};
                file.write(reinterpret_cast<const char*>(identifier), 12);
                // Header (simplified)
                struct KTX2Header {
                    uint32_t vkFormat = 0;
                    uint32_t typeSize = 1;
                    uint32_t pixelWidth;
                    uint32_t pixelHeight;
                    uint32_t pixelDepth = 0;
                    uint32_t layerCount = 0;
                    uint32_t faceCount = 1;
                    uint32_t levelCount = 1;
                    uint32_t supercompressionScheme = 0;
                } hdr;
                hdr.pixelWidth = static_cast<uint32_t>(width);
                hdr.pixelHeight = static_cast<uint32_t>(height);
                hdr.vkFormat = format; // e.g., VK_FORMAT_BC3_UNORM_BLOCK = 133
                file.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
                // Level index
                uint64_t levelByteOffset = sizeof(identifier) + sizeof(hdr) + 3*8 + 8; // simplified
                uint64_t levelByteLength = compressed_data.size();
                uint64_t uncompressedByteLength = 0;
                file.write(reinterpret_cast<const char*>(&levelByteOffset), 8);
                file.write(reinterpret_cast<const char*>(&levelByteLength), 8);
                file.write(reinterpret_cast<const char*>(&uncompressedByteLength), 8);
                // Data
                file.write(reinterpret_cast<const char*>(compressed_data.data()), compressed_data.size());
            }

            // --------------------------------------------------------------------
            // Texture compression quality metrics
            // --------------------------------------------------------------------
            inline double compute_psnr(const ImageU8& original, const ImageU8& compressed)
            {
                if (original.shape() != compressed.shape())
                    XTENSOR_THROW(std::invalid_argument, "PSNR: shape mismatch");
                double mse = 0.0;
                size_t n = original.size();
                for (size_t i = 0; i < n; ++i)
                {
                    int diff = static_cast<int>(original.flat(i)) - static_cast<int>(compressed.flat(i));
                    mse += diff * diff;
                }
                mse /= n;
                if (mse == 0) return std::numeric_limits<double>::infinity();
                return 20.0 * std::log10(255.0 / std::sqrt(mse));
            }

            inline double compute_ssim(const ImageU8& original, const ImageU8& compressed, double K1=0.01, double K2=0.03)
            {
                // Simplified SSIM on luminance only
                if (original.dimension() == 3)
                {
                    auto gray_orig = image::color::rgb2gray(original);
                    auto gray_comp = image::color::rgb2gray(compressed);
                    double mu_x = xt::mean(gray_orig)();
                    double mu_y = xt::mean(gray_comp)();
                    double var_x = xt::var(gray_orig)();
                    double var_y = xt::var(gray_comp)();
                    double cov = 0.0;
                    size_t n = gray_orig.size();
                    for (size_t i = 0; i < n; ++i)
                        cov += (gray_orig.flat(i) - mu_x) * (gray_comp.flat(i) - mu_y);
                    cov /= n;
                    double L = 255.0;
                    double C1 = (K1 * L) * (K1 * L);
                    double C2 = (K2 * L) * (K2 * L);
                    double ssim = (2.0 * mu_x * mu_y + C1) * (2.0 * cov + C2) /
                                  ((mu_x*mu_x + mu_y*mu_y + C1) * (var_x + var_y + C2));
                    return ssim;
                }
                return 0.0;
            }

        } // namespace texture

        // Bring texture functions into xt namespace
        using texture::bc::DXT1Block;
        using texture::bc::DXT5Block;
        using texture::bc::compress_bc1_block;
        using texture::bc::compress_bc3;
        using texture::bc::decompress_bc3;
        using texture::generate_mipmaps;
        using texture::height_to_normal;
        using texture::write_dds;
        using texture::pack_rectangles;
        using texture::build_texture_atlas;
        using texture::write_ktx2;
        using texture::compute_psnr;
        using texture::compute_ssim;

    } // XTENSOR_INLINE_NAMESPACE
} // namespace xt

#endif // XTENSOR_XTEXTURE_COMPRESSOR_HPP

// image/xtexture_compressor.hpp