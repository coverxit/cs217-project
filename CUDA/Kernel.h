#pragma once

__global__ void CompressKernel(const uint8_t* deviceInBuf, int64_t inSize,
    uint8_t* deviceOutBuf, int64_t* deviceOutSize,
    CompressFlagBlock* deviceFlagOut, int64_t* deviceFlagSize);

__global__ void DecompressKernel(CompressFlagBlock* deviceFlagIn, int64_t nFlagBlocks,
    const uint8_t* deviceInBuf, uint8_t* deviceOutBuf);
