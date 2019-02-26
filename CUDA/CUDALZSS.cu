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
    do { cudaError_t ret = (op); if ((ret) != cudaSuccess) _gerror((ret), (msg), __LINE__); } while (false)

inline void _gerror(cudaError_t cudaError, const char* msg, int lineNo) {
    fprintf(stderr, "[%s:%d] %s, cudaError = %d\n", lineNo, __FILE__, msg, cudaError);
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
    cudaCheckError(cudaMalloc((void**) &deviceInBuf, inSize), "Failed to allocate device memory");
    cudaCheckError(cudaMalloc((void**) &deviceOutBuf, outSize), "Failed to allocate device memory");
    cudaCheckError(cudaMalloc((void**) &deviceOutSize, sizeof(int)), "Failed to allocate device memory");
    cudaCheckError(cudaMalloc((void**) &deviceFlagSize, sizeof(int)), "Failed to allocate device memory");

    printf("Copying data from host to device...\n");
    cudaCheckError(cudaMemcpy(deviceInBuf, inBuf, inSize, cudaMemcpyHostToDevice), "Failed to copy memory to device");
    cudaCheckError(cudaMemset(deviceFlagOut, 0, sizeof(CompressFlagBlock) * nFlagBlocks),
        "Failed to set device memory");

    timer.begin();
    //CompressKernel<<<,>>>();
    auto elapsed = timer.end();

    printf("Copying data from device to host...\n");
    cudaCheckError(cudaMemcpy(&outSize, deviceOutSize, sizeof(int), cudaMemcpyDeviceToHost),
        "Failed to copy memory to host");
    cudaCheckError(cudaMemcpy(&flagSize, deviceFlagSize, sizeof(int), cudaMemcpyDeviceToHost),
        "Failed to copy memory to host");

    cudaCheckError(cudaMemcpy(outBuf, deviceOutBuf, outSize, cudaMemcpyDeviceToHost), "Failed to copy memory to host");
    cudaCheckError(cudaMemcpy(flagOut, deviceFlagOut, sizeof(CompressFlagBlock) * nFlagBlocks, cudaMemcpyDeviceToHost),
        "Failed to copy memory to host");
    
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
