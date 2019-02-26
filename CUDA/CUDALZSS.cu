#include <utility>

#include <memory.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../Settings.h"

#include "../LZSSInterface.h"
#include "../TimerHelper.hpp"

#include "../MatchHelper/MatchHelper.h"
#include "CUDALZSS.h"

#define cudaCheckError(op, msg) \
    do { cudaError_t ret = (op); if ((ret) != cudaSuccess) _gerror((ret), (msg)); } while (false)

inline void _gerror(cudaError_t cudaError, const char* msg) {
    fprintf(stderr, "%s, cudaError = %d\n", msg, cudaError);
    exit(-1);
}

__global__ void CompressKernel(const uint8_t* deviceInBuf, int inSize,
    uint8_t* deviceOutBuf, int* deviceOutSize, CompressFlagBlock* deviceFlagOut, int nFlagBlocks, int* deviceFlagSize) {
    
}

std::pair<bool, double> CUDALZSS::compress(const uint8_t* inBuf, int inSize,
    uint8_t* outBuf, int& outSize,
    CompressFlagBlock* flagOut, int nFlagBlocks, int& flagSize)
{
    Timer timer(false);

    uint8_t *deviceInBuf, *deviceOutBuf;
    CompressFlagBlock *deviceFlagOut;
    int *deviceOutSize, *deviceFlagSize;

    printf("Allocating device variables...\n");
    cudaCheckError(cudaMalloc((void**) &deviceInBuf, inSize), "Failed to allocate deviceInBuf");

    cudaCheckError(cudaMalloc((void**) &deviceOutBuf, outSize), "Failed to allocate deviceOutBuf");
    cudaCheckError(cudaMalloc((void**) &deviceOutSize, sizeof(int)), "Failed to allocate deviceOutSize");

    cudaCheckError(cudaMalloc((void**) &deviceFlagOut, sizeof(CompressFlagBlock) * nFlagBlocks), 
        "Failed to allocate deviceFlagOut");
    cudaCheckError(cudaMalloc((void**) &deviceFlagSize, sizeof(int)), "Failed to allocate deviceFlagSize");

    printf("Copying data from host to device...\n");
    cudaCheckError(cudaMemcpy(deviceInBuf, inBuf, inSize, cudaMemcpyHostToDevice), 
        "Failed to copy deviceInBuf to device");
    cudaCheckError(cudaMemset(deviceFlagOut, 0, sizeof(CompressFlagBlock) * nFlagBlocks), 
        "Failed to set deviceFlagOut to 0");

    timer.begin();
    //CompressKernel<<<,>>>();
    auto elapsed = timer.end();

    printf("Copying data from device to host...\n");
    cudaCheckError(cudaMemcpy(&outSize, deviceOutSize, sizeof(int), cudaMemcpyDeviceToHost),
        "Failed to copy deviceOutSize to host");
    cudaCheckError(cudaMemcpy(&flagSize, deviceFlagSize, sizeof(int), cudaMemcpyDeviceToHost),
        "Failed to copy deviceFlagSize to host");

    cudaCheckError(cudaMemcpy(outBuf, deviceOutBuf, outSize, cudaMemcpyDeviceToHost), 
        "Failed to copy deviceOutBuf to host");
    cudaCheckError(cudaMemcpy(flagOut, deviceFlagOut, sizeof(CompressFlagBlock) * nFlagBlocks, cudaMemcpyDeviceToHost),
        "Failed to copy deviceFlagOut to host");
    
    return std::make_pair(true, elapsed);
}

std::pair<bool, double> CUDALZSS::decompress(CompressFlagBlock* flagIn, int nFlagBlocks,
    const uint8_t* inBuf, int inSize, uint8_t* outBuf)
{
    Timer timer(false);

    timer.begin();
    //DecompressKernel<<<,>>>();
    auto elapsed = timer.end();

    return std::make_pair(true, elapsed);
}
