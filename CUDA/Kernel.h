#pragma once

__global__ void CompressKernel(const uint8_t* deviceInBuf, int inSize,
    uint8_t* deviceOutBuf, int* deviceOutSize,
    CompressFlagBlock* deviceFlagOut, int* deviceFlagSize);

__global__ void DecompressKernel(CompressFlagBlock* deviceFlagIn,
    const uint8_t* deviceInBuf, int nFlagBlocks, uint8_t* deviceOutBuf);
