#pragma once

constexpr int DefaultMagic = 0x5A545247; // GRTZ
constexpr int MaxFileSize = 2147483647;  // 32 bit signed integer, maximum 2 GB

typedef uint16_t PairType;               // 2 bytes
constexpr int PairOffsetBits = 11;       // 2 KB search window
constexpr int PairLengthBits = 5;        // Maximum replacement length = 32

constexpr int WindowSize = 1 << PairOffsetBits; 
constexpr int MaxEncodeLength = 1 << PairLengthBits;

constexpr int DataBlockSize = 4096;      // Divide data into 4096-byte blocks for parallel processing
constexpr int GPUBlockSize = 256;        // 256 threads per block
constexpr int NumOfCUDAStream = 16;      // Number of CUDA streams

constexpr int ReplaceThreshold = 3;      // At least 3 bytes match before we perform replacement
