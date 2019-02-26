#include <utility>

#include <memory.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../Settings.h"

#include "../BitHelper.h"
#include "../LZSSInterface.h"
#include "../TimerHelper.hpp"

#include "../MatchHelper/MatchHelper.h"
#include "CUDALZSS.h"
#include "Kernel.h"

#define cudaCheckError(op, msg)    \
    do {                           \
        cudaError_t ret = (op);    \
        if ((ret) != cudaSuccess)  \
            _gerror((ret), (msg)); \
    } while (false)

inline void _gerror(cudaError_t cudaError, const char* msg)
{
    fprintf(stderr, "%s, cudaError = %d\n", msg, cudaError);
    exit(-1);
}

std::pair<bool, double> CUDALZSS::compress(const uint8_t* inBuf, int inSize,
    uint8_t* outBuf, int& outSize,
    CompressFlagBlock* flagOut, int nFlagBlocks, int& flagSize)
{
    Timer timer(false);

    uint8_t *deviceInBuf, *deviceOutBuf;
    CompressFlagBlock* deviceFlagOut;
    int *deviceOutSize, *deviceFlagSize;

    // Allocate ----------------------------------
    printf("Allocating device variables... ");
    fflush(stdout);

    timer.begin();
    cudaCheckError(cudaMalloc((void**)&deviceInBuf, inSize), "Failed to allocate deviceInBuf");

    cudaCheckError(cudaMalloc((void**)&deviceOutBuf, outSize), "Failed to allocate deviceOutBuf");
    cudaCheckError(cudaMalloc((void**)&deviceOutSize, sizeof(int)), "Failed to allocate deviceOutSize");

    cudaCheckError(cudaMalloc((void**)&deviceFlagOut, sizeof(CompressFlagBlock) * nFlagBlocks),
        "Failed to allocate deviceFlagOut");
    cudaCheckError(cudaMalloc((void**)&deviceFlagSize, sizeof(int)), "Failed to allocate deviceFlagSize");
    cudaDeviceSynchronize();
    printf("%.6fs\n", timer.end());

    // Copy: host to device -----------------------
    printf("Copying data from host to device... ");
    fflush(stdout);

    timer.begin();
    cudaCheckError(cudaMemcpy(deviceInBuf, inBuf, inSize, cudaMemcpyHostToDevice),
        "Failed to copy inBuf to device");
    cudaCheckError(cudaMemset(deviceFlagOut, 0, sizeof(CompressFlagBlock) * nFlagBlocks),
        "Failed to set deviceFlagOut to 0");
    cudaDeviceSynchronize();
    printf("%.6fs\n", timer.end());

    // Launch kernel ------------------------------
    printf("Launching kernel...\n");
    fflush(stdout);

    timer.begin();
    CompressKernel<<<nFlagBlocks, GPUBlockSize>>>(deviceInBuf, inSize,
        deviceOutBuf, deviceOutSize,
        deviceFlagOut, nFlagBlocks, deviceFlagSize);
    cudaCheckError(cudaDeviceSynchronize(), "Failed to launch kernel");
    auto elapsed = timer.end();

    // Copy: device to host -----------------------
    printf("Copying data from device to host... ");
    fflush(stdout);

    timer.begin();
    cudaCheckError(cudaMemcpy(&outSize, deviceOutSize, sizeof(int), cudaMemcpyDeviceToHost),
        "Failed to copy deviceOutSize to host");
    cudaCheckError(cudaMemcpy(&flagSize, deviceFlagSize, sizeof(int), cudaMemcpyDeviceToHost),
        "Failed to copy deviceFlagSize to host");

    cudaCheckError(cudaMemcpy(outBuf, deviceOutBuf, inSize, cudaMemcpyDeviceToHost),
        "Failed to copy deviceOutBuf to host");
    cudaCheckError(cudaMemcpy(flagOut, deviceFlagOut, sizeof(CompressFlagBlock) * nFlagBlocks, cudaMemcpyDeviceToHost),
        "Failed to copy deviceFlagOut to host");
    cudaDeviceSynchronize();
    printf("%.6fs\n", timer.end());
    fflush(stdout);

    cudaFree(deviceInBuf);
    cudaFree(deviceOutBuf);
    cudaFree(deviceOutSize);
    cudaFree(deviceFlagOut);
    cudaFree(deviceFlagSize);

    return std::make_pair(true, elapsed);
}

std::pair<bool, double> CUDALZSS::decompress(CompressFlagBlock* flagIn, int nFlagBlocks,
    const uint8_t* inBuf, int inSize, uint8_t* outBuf, int outSize)
{
    Timer timer(false);

    uint8_t *deviceInBuf, *deviceOutBuf, *deviceFlagIn;

    // Allocate ----------------------------------
    printf("Allocating device variables... ");
    fflush(stdout);

    timer.begin();
    cudaCheckError(cudaMalloc((void**)&deviceInBuf, inSize), "Failed to allocate deviceInBuf");
    cudaCheckError(cudaMalloc((void**)&deviceOutBuf, outSize), "Failed to allocate deviceOutBuf");
    cudaCheckError(cudaMalloc((void**)&deviceFlagIn, sizeof(CompressFlagBlock) * nFlagBlocks),
        "Failed to allocate deviceFlagIn");
    cudaDeviceSynchronize();
    printf("%.6fs\n", timer.end());

    // Copy: host to device -----------------------
    printf("Copying data from host to device... ");
    fflush(stdout);

    timer.begin();
    cudaCheckError(cudaMemcpy(deviceInBuf, inBuf, inSize, cudaMemcpyHostToDevice),
        "Failed to copy inBuf to device");
    cudaCheckError(cudaMemcpy(deviceFlagIn, flagIn, sizeof(CompressFlagBlock) * nFlagBlocks, cudaMemcpyHostToDevice),
        "Failed to copy flagIn to device");
    cudaDeviceSynchronize();
    printf("%.6fs\n", timer.end());

    // Launch kernel ------------------------------
    printf("Launching kernel...\n");
    fflush(stdout);

    timer.begin();
    auto dimGrid = (nFlagBlocks - 1) / GPUBlockSize + 1;
    DecompressKernel<<<dimGrid, GPUBlockSize>>>(deviceFlagIn, nFlagBlocks, deviceInBuf, deviceOutBuf);
    cudaCheckError(cudaDeviceSynchronize(), "Failed to launch kernel");
    auto elapsed = timer.end();

    // Copy: device to host -----------------------
    printf("Copying data from device to host... ");
    fflush(stdout);

    timer.begin();
    cudaCheckError(cudaMemcpy(outBuf, deviceOutBuf, outSize, cudaMemcpyDeviceToHost),
        "Failed to copy deviceOutBuf to host");
    cudaDeviceSynchronize();
    printf("%.6fs\n", timer.end());
    fflush(stdout);

    cudaFree(deviceInBuf);
    cudaFree(deviceOutBuf);
    cudaFree(deviceFlagIn);

    return std::make_pair(true, elapsed);
}
