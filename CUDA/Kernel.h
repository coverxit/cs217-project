#pragma once

__global__ void CompressKernel(const uint8_t* deviceInBuf, int inSize,
    uint8_t* deviceOutBuf, int* deviceOutSize,
    CompressFlagBlock* deviceFlagOut, int nFlagBlocks, int* deviceFlagSize,
    int* deviceNumBlocksDone);
