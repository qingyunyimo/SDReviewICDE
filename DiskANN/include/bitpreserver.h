#pragma once
#include <memory>
#include <vector>   
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <immintrin.h>
class BitPreserver{
private:
    int d;  // 维度
    int m;  // 每个维度保留的bit数
    size_t chopsize; // 缓冲区大小

public:

    BitPreserver(int dimensions, int bits_per_dimension) 
        : d(dimensions), m(bits_per_dimension) {
        // 计算需要的字节数
        size_t total_bits = static_cast<size_t>(d) * m;
        chopsize = (total_bits + 7) / 8;  // 向上取整到字节
    }

    void setNewParams(int dimensions, int bits_per_dimension) {
        d = dimensions;
        m = bits_per_dimension;
        size_t total_bits = static_cast<size_t>(d) * m;
        chopsize = (total_bits + 7) / 8;  // 向上取整到字节
    }

    // 将浮点数数据的前m位保存到缓冲区
    std::unique_ptr<char[]> saveFloatBits(const std::vector<float>& data) {
        if (data.size() != static_cast<size_t>(d)) {
            throw std::invalid_argument("数据维度不匹配");
        }
        
        // 创建缓冲区
        std::unique_ptr<char[]> middle_buf = std::make_unique<char[]>(chopsize);
        std::memset(middle_buf.get(), 0, chopsize);  // 初始化为0
        
        // 遍历每个维度
        for (int dim = 0; dim < d; ++dim) {
            float value = data[dim];
            
            // 将float转换为uint32_t以获取bit表示
            uint32_t int_value;
            std::memcpy(&int_value, &value, sizeof(float));
            
            // 提取前m位（从最高有效位开始）
            uint32_t mask = 0x80000000;  // 最高位为1
            int bit_pos = 0;  // 当前维度在当前字节中的bit位置
            
            for (int bit = 0; bit < m; ++bit) {
                // 提取当前bit
                bool bit_value = (int_value & mask) != 0;
                
                // 计算在缓冲区中的位置
                size_t global_bit_pos = static_cast<size_t>(dim) * m + bit;
                size_t byte_pos = global_bit_pos / 8;
                int bit_offset = 7 - (global_bit_pos % 8);  // 从左到右存储
                
                // 设置bit
                if (bit_value) {
                    middle_buf[byte_pos] |= (1 << bit_offset);
                }
                
                // 移动到下一个bit
                mask >>= 1;
            }
        }
        
        return middle_buf;
    }

    // 将浮点数数据的前m位保存到缓冲区
    std::unique_ptr<char[]> saveFloatBits(const float* data, unsigned data_dim) {
        if (data_dim != static_cast<size_t>(d)) {
            throw std::invalid_argument("数据维度不匹配");
        }
        
        // 创建缓冲区
        std::unique_ptr<char[]> middle_buf = std::make_unique<char[]>(chopsize);
        std::memset(middle_buf.get(), 0, chopsize);  // 初始化为0
        
        // 遍历每个维度
        for (int dim = 0; dim < d; ++dim) {
            float value = data[dim];
            
            // 将float转换为uint32_t以获取bit表示
            uint32_t int_value;
            std::memcpy(&int_value, &value, sizeof(float));
            
            // 提取前m位（从最高有效位开始）
            uint32_t mask = 0x80000000;  // 最高位为1
            int bit_pos = 0;  // 当前维度在当前字节中的bit位置
            
            for (int bit = 0; bit < m; ++bit) {
                // 提取当前bit
                bool bit_value = (int_value & mask) != 0;
                
                // 计算在缓冲区中的位置
                size_t global_bit_pos = static_cast<size_t>(dim) * m + bit;
                size_t byte_pos = global_bit_pos / 8;
                int bit_offset = 7 - (global_bit_pos % 8);  // 从左到右存储
                
                // 设置bit
                if (bit_value) {
                    middle_buf[byte_pos] |= (1 << bit_offset);
                }
                
                // 移动到下一个bit
                mask >>= 1;
            }
        }
        
        return middle_buf;
    }

    //小端存储
    std::unique_ptr<char[]> saveFloatHigherBitsLD16bit(const float* data, unsigned data_dim) {
        assert(m == 16);
        std::unique_ptr<char[]> middle_buf = std::make_unique<char[]>(chopsize);
        std::memset(middle_buf.get(), 0, chopsize);  // 初始化为0

        for (int dim = 0; dim < d; ++dim) {
            uint32_t int_value;
            std::memcpy(&int_value, &data[dim], sizeof(float));
            uint16_t val = static_cast<uint16_t>(int_value >> 16); // 取高16位
            std::memcpy(&middle_buf[dim * 2], &val, sizeof(uint16_t));
        }
        return middle_buf;
    }

    // 将浮点数数据的前m位保存到缓冲区
    std::unique_ptr<char[]> saveFloatLowerBits(const float* data, unsigned data_dim) {
        if (data_dim != static_cast<size_t>(d)) {
            throw std::invalid_argument("数据维度不匹配");
        }
        
        // 创建缓冲区
        std::unique_ptr<char[]> middle_buf = std::make_unique<char[]>(chopsize);
        std::memset(middle_buf.get(), 0, chopsize);  // 初始化为0
        
        // 遍历每个维度
        for (int dim = 0; dim < d; ++dim) {
            float value = data[dim];
            
            // 将float转换为uint32_t以获取bit表示
            uint32_t int_value;
            std::memcpy(&int_value, &value, sizeof(float));
            int_value = int_value << (32 - m);
            
            // 提取前m位（从最高有效位开始）
            uint32_t mask = 0x80000000;  // 最高位为1
            int bit_pos = 0;  // 当前维度在当前字节中的bit位置
            
            for (int bit = 0; bit < m; ++bit) {
                // 提取当前bit
                bool bit_value = (int_value & mask) != 0;
                
                // 计算在缓冲区中的位置
                size_t global_bit_pos = static_cast<size_t>(dim) * m + bit;
                size_t byte_pos = global_bit_pos / 8;
                int bit_offset = 7 - (global_bit_pos % 8);  // 从左到右存储
                
                // 设置bit
                if (bit_value) {
                    middle_buf[byte_pos] |= (1 << bit_offset);
                }
                
                // 移动到下一个bit
                mask >>= 1;
            }
        }
        
        return middle_buf;
    }

    //小端存储
    std::unique_ptr<char[]> saveFloatLowerBitsLD16bit(const float* data, unsigned data_dim) {
        assert(m == 16);
        std::unique_ptr<char[]> middle_buf = std::make_unique<char[]>(chopsize);
        std::memset(middle_buf.get(), 0, chopsize);  // 初始化为0

        for (int dim = 0; dim < d; ++dim) {
            uint32_t int_value;
            std::memcpy(&int_value, &data[dim], sizeof(float));
            uint16_t val = static_cast<uint16_t>(int_value & 0xFFFF); // 取低16位
            std::memcpy(&middle_buf[dim * 2], &val, sizeof(uint16_t));
        }
        return middle_buf;
    }

    // 从缓冲区恢复浮点数数据（近似值）
    std::vector<float> loadFloatBits(const char* middle_buf) {
        std::vector<float> result(d, 0.0f);
        
        for (int dim = 0; dim < d; ++dim) {
            // 重建uint32_t
            uint32_t int_value = 0;
            uint32_t mask = 0x80000000;  // 最高位为1
            
            for (int bit = 0; bit < m; ++bit) {
                // 计算在缓冲区中的位置
                size_t global_bit_pos = static_cast<size_t>(dim) * m + bit;
                size_t byte_pos = global_bit_pos / 8;
                int bit_offset = 7 - (global_bit_pos % 8);  // 从左到右存储
                
                // 读取bit
                bool bit_value = (middle_buf[byte_pos] & (1 << bit_offset)) != 0;
                
                // 设置bit
                if (bit_value) {
                    int_value |= mask;
                }
                
                // 移动到下一个bit
                mask >>= 1;
            }
            
            // 将剩余的bits设为0
            //for (int bit = m; bit < 32; ++bit) {
            //    mask >>= 1;  // 将剩余位设为0
            //}
            
            // 转换为float
            float value;
            std::memcpy(&value, &int_value, sizeof(float));
            result[dim] = value;
        }
        
        return result;
    }

    // 从缓冲区恢复浮点数数据（近似值）
    void loadFloatBits(const char* middle_buf, std::vector<float> &result) {
        
        for (int dim = 0; dim < d; ++dim) {
            // 重建uint32_t
            uint32_t int_value = 0;
            uint32_t mask = 0x80000000;  // 最高位为1
            
            for (int bit = 0; bit < m; ++bit) {
                // 计算在缓冲区中的位置
                size_t global_bit_pos = static_cast<size_t>(dim) * m + bit;
                size_t byte_pos = global_bit_pos / 8;
                int bit_offset = 7 - (global_bit_pos % 8);  // 从左到右存储
                
                // 读取bit
                bool bit_value = (middle_buf[byte_pos] & (1 << bit_offset)) != 0;
                
                // 设置bit
                if (bit_value) {
                    int_value |= mask;
                }
                
                // 移动到下一个bit
                mask >>= 1;
            }
            
            // 将剩余的bits设为0
            //for (int bit = m; bit < 32; ++bit) {
            //    mask >>= 1;  // 将剩余位设为0
            //}
            
            // 转换为float
            float value;
            std::memcpy(&value, &int_value, sizeof(float));
            result[dim] = value;
        }
    }

    void loadFloatFromHigh16BitsAVX512(const char* input, float* output) {
        assert(m == 16);
        const uint16_t* data_16bit = reinterpret_cast<const uint16_t*>(input);
        constexpr size_t SIMD_WIDTH = 16;  // 一次处理16个float
        
        size_t i = 0;
        
        for (; i + SIMD_WIDTH <= d; i += SIMD_WIDTH) {
            // 加载16个16位数据到256位寄存器
            __m256i v16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&data_16bit[i]));
            
            // 直接使用vpmovzxwd将16位扩展到32位（256->512）
            __m512i v32 = _mm512_cvtepu16_epi32(v16);
            
            // 左移16位到高位
            v32 = _mm512_slli_epi32(v32, 16);
            
            // 转换为浮点数
            __m512 float_vec = _mm512_castsi512_ps(v32);
            
            // 存储结果
            _mm512_storeu_ps(&output[i], float_vec);
        }
        
        // 处理剩余部分（标量）
        for (; i < d; ++i) {
            std::cout<<"run in here"<<std::endl;
            uint32_t int_value = static_cast<uint32_t>(data_16bit[i]) << 16;
            std::memcpy(&output[i], &int_value, sizeof(float));
        }
    }

    void loadFloatFromLow16BitsAVX512(const char* input, float* output){
        assert(m == 16);
    
        const uint16_t* data_16bit = reinterpret_cast<const uint16_t*>(input);
        constexpr size_t SIMD_WIDTH = 16;
        
        size_t i = 0;
        // 主要循环：一次处理16个float
        for (; i + SIMD_WIDTH <= d; i += SIMD_WIDTH) {
            // 加载16个16位数据
            __m256i v16 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(&data_16bit[i]));
            
            // 零扩展到32位
            __m512i v32 = _mm512_cvtepu16_epi32(v16);
            
            // 注意：原始代码将数据放在低16位，所以这里不需要移位
            // 直接将16位数据作为float的低16位，高16位为0
            
            // 转换为float
            __m512 float_vec = _mm512_castsi512_ps(v32);
            
            // 存储
            _mm512_storeu_ps(&output[i], float_vec);
        }
        
        // 标量处理尾部
        for (; i < d; ++i) {
            uint32_t int_value = static_cast<uint32_t>(data_16bit[i]);
            std::memcpy(&output[i], &int_value, sizeof(float));
        }
    }

    void restoreFloatFromHighLow16BitsAVX512(
        const char* high_bits_buf,   // 高16位缓冲区
        const char* low_bits_buf,    // 低16位缓冲区
        std::vector<float>& result) {

        const uint16_t* high_bits = reinterpret_cast<const uint16_t*>(high_bits_buf);
        const uint16_t* low_bits = reinterpret_cast<const uint16_t*>(low_bits_buf);
        
        constexpr size_t SIMD_WIDTH = 16;  // AVX512一次处理16个float
        
        size_t i = 0;
        // 主循环：一次处理16个float
        for (; i + SIMD_WIDTH <= static_cast<size_t>(d); i += SIMD_WIDTH) {
            // 1. 加载高16位数据（16个uint16_t）
            __m256i high_16 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(&high_bits[i]));
            
            // 2. 加载低16位数据（16个uint16_t）
            __m256i low_16 = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(&low_bits[i]));
            
            // 3. 将高低16位分别零扩展到32位
            __m512i high_32 = _mm512_cvtepu16_epi32(high_16);
            __m512i low_32 = _mm512_cvtepu16_epi32(low_16);
            
            // 4. 高16位移到32位的高位（左移16位）
            high_32 = _mm512_slli_epi32(high_32, 16);
            
            // 5. 低16位保持在低位（使用掩码确保高16位为0）
            __m512i low_mask = _mm512_set1_epi32(0x0000FFFF);
            low_32 = _mm512_and_si512(low_32, low_mask);
            
            // 6. 合并高低位：高16位 | 低16位
            __m512i merged = _mm512_or_si512(high_32, low_32);
            
            // 7. 转换为float
            __m512 float_vec = _mm512_castsi512_ps(merged);
            
            // 8. 存储结果
            _mm512_storeu_ps(&result[i], float_vec);
        }
        
        // 处理剩余部分（不足16个）
        for (; i < static_cast<size_t>(d); ++i) {
            uint32_t high_part = static_cast<uint32_t>(high_bits[i]) << 16;
            uint32_t low_part = static_cast<uint32_t>(low_bits[i]) & 0x0000FFFF;
            uint32_t merged = high_part | low_part;
            std::memcpy(&result[i], &merged, sizeof(float));
        }
    }

    std::vector<float> loadFloatLowerBits(const char* middle_buf){
        std::vector<float> result(d, 0);
        
        for (int dim = 0; dim < d; ++dim) {
            // 重建uint32_t
            uint32_t int_value = 0;
            uint32_t mask = 0x80000000;  // 最高位为1
            
            for (int bit = 0; bit < m; ++bit) {
                // 计算在缓冲区中的位置
                size_t global_bit_pos = static_cast<size_t>(dim) * m + bit;
                size_t byte_pos = global_bit_pos / 8;
                int bit_offset = 7 - (global_bit_pos % 8);  // 从左到右存储
                
                // 读取bit
                bool bit_value = (middle_buf[byte_pos] & (1 << bit_offset)) != 0;
                
                // 设置bit
                if (bit_value) {
                    int_value |= mask;
                }
                
                // 移动到下一个bit
                mask >>= 1;
            }
            
            // 将剩余的bits设为0
            //for (int bit = m; bit < 32; ++bit) {
            //    mask >>= 1;  // 将剩余位设为0
            //}
            
            int_value = int_value >> (32 - m);            
            std::memcpy(&result[dim], &int_value, sizeof(float));
        }
        
        return result;
    }

    void loadFloatLowerBits(const char* middle_buf, std::vector<float> &result) {
        
        for (int dim = 0; dim < d; ++dim) {
            // 重建uint32_t
            uint32_t int_value = 0;
            uint32_t mask = 0x80000000;  // 最高位为1
            
            for (int bit = 0; bit < m; ++bit) {
                // 计算在缓冲区中的位置
                size_t global_bit_pos = static_cast<size_t>(dim) * m + bit;
                size_t byte_pos = global_bit_pos / 8;
                int bit_offset = 7 - (global_bit_pos % 8);  // 从左到右存储
                
                // 读取bit
                bool bit_value = (middle_buf[byte_pos] & (1 << bit_offset)) != 0;
                
                // 设置bit
                if (bit_value) {
                    int_value |= mask;
                }
                
                // 移动到下一个bit
                mask >>= 1;
            }
            
            // 将剩余的bits设为0
            //for (int bit = m; bit < 32; ++bit) {
            //    mask >>= 1;  // 将剩余位设为0
            //}
            int_value = int_value >> (32 - m);
            std::memcpy(&result[dim], &int_value, sizeof(float));
        }
        
    }
    
    // 获取缓冲区大小
    size_t getChopSize() const {
        return chopsize;
    }
    
    // 打印缓冲区内容（用于调试）
    void printBuffer(const char* buffer) const {
        std::cout << "缓冲区内容 (" << chopsize << " 字节):" << std::endl;
        for (size_t i = 0; i < chopsize; ++i) {
            std::cout << "字节 " << i << ": ";
            for (int bit = 7; bit >= 0; --bit) {
                std::cout << ((buffer[i] & (1 << bit)) ? '1' : '0');
            }
            std::cout << std::endl;
        }
    }

};