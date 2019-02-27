#include <algorithm>
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
    fprintf(stderr, "%s, CUDA Error (%d): %s\n", msg, cudaError, cudaGetErrorString(cudaError));
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
    int *hostOutSize, *hostFlagSize;
    cudaStream_t *cudaStreams;

    auto numOfStreams = std::min(nFlagBlocks, NumOfCUDAStream);
    auto blocksPerStream = (nFlagBlocks - 1) / numOfStreams + 1;
    auto alignedStreamSize = blocksPerStream * DataBlockSize;

    // Create stream -----------------------------
    cudaStreams = new cudaStream_t[numOfStreams];
    for (int i = 0; i < numOfStreams; ++i) {
        cudaCheckError(cudaStreamCreate(&cudaStreams[i]), "Failed to create CUDA streams");
    }

    // Allocate ----------------------------------
    printf("Allocating device variables... ");
    fflush(stdout);

    timer.begin();
    cudaCheckError(cudaMalloc((void**)&deviceInBuf, inSize), "Failed to allocate deviceInBuf");

    hostOutSize = new int[numOfStreams];
    hostFlagSize = new int[numOfStreams];
    cudaCheckError(cudaMalloc((void**)&deviceOutBuf, inSize), "Failed to allocate deviceOutBuf");
    cudaCheckError(cudaMalloc((void**)&deviceOutSize, numOfStreams * sizeof(int)), "Failed to allocate deviceOutSize");

    cudaCheckError(cudaMalloc((void**)&deviceFlagOut, sizeof(CompressFlagBlock) * nFlagBlocks),
        "Failed to allocate deviceFlagOut");
    cudaCheckError(cudaMalloc((void**)&deviceFlagSize, numOfStreams * sizeof(int)), "Failed to allocate deviceFlagSize");
    cudaDeviceSynchronize();
    printf("%.6fs\n", timer.end());

    // Copy: host to device -----------------------
    printf("Copying data from host to device... ");
    fflush(stdout);

    timer.begin();

    for (int i = 0; i < numOfStreams; ++i) {
        auto streamSize = std::min(alignedStreamSize, inSize - i * alignedStreamSize);
        auto numOfBlock = std::min(blocksPerStream, nFlagBlocks - i * blocksPerStream);

        cudaCheckError(cudaMemcpyAsync(deviceInBuf + i * alignedStreamSize, 
            inBuf + i * alignedStreamSize, streamSize, 
            cudaMemcpyHostToDevice, cudaStreams[i]), 
            "Failed to copy inBuf to device");

        cudaCheckError(cudaMemsetAsync(deviceFlagOut + i * blocksPerStream, 0, 
            sizeof(CompressFlagBlock) * numOfBlock, cudaStreams[i]),
            "Failed to set deviceFlagOut to 0");
    }

    cudaCheckError(cudaMemset(deviceOutSize, 0, sizeof(int) * numOfStreams), "Failed to set deviceOutSize to 0");
    cudaCheckError(cudaMemset(deviceFlagSize, 0, sizeof(int) * numOfStreams), "Failed to set deviceFlagSize to 0");
    cudaDeviceSynchronize();
    printf("%.6fs\n", timer.end());

    // Launch kernel ------------------------------
    printf("Launching kernel... ");
    fflush(stdout);

    timer.begin();

    for (int i = 0; i < numOfStreams; ++i) {
        auto streamSize = std::min(alignedStreamSize, inSize - i * alignedStreamSize);
        auto numOfBlock = std::min(blocksPerStream, nFlagBlocks - i * blocksPerStream);

        CompressKernel<<<numOfBlock, GPUBlockSize, 0, cudaStreams[i]>>>(
            deviceInBuf + i * alignedStreamSize, streamSize,
            deviceOutBuf + i * alignedStreamSize, &deviceOutSize[i],
            deviceFlagOut + i * blocksPerStream, &deviceFlagSize[i]);
    }

    cudaCheckError(cudaDeviceSynchronize(), "Failed to launch kernel");
    auto elapsed = timer.end();
    printf("%.6fs\n", elapsed);

    // Copy: device to host -----------------------
    printf("Copying data from device to host... ");
    fflush(stdout);

    timer.begin();
    
    for (int i = 0; i < numOfStreams; ++i) {
        auto streamSize = std::min(alignedStreamSize, inSize - i * alignedStreamSize);
        auto numOfBlock = std::min(blocksPerStream, nFlagBlocks - i * blocksPerStream);

        cudaCheckError(cudaMemcpyAsync(outBuf + i * alignedStreamSize, 
            deviceOutBuf + i * alignedStreamSize, streamSize, 
            cudaMemcpyDeviceToHost, cudaStreams[i]), 
            "Failed to copy deviceOutBuf to host");

        cudaCheckError(cudaMemcpyAsync(flagOut + i * blocksPerStream,
            deviceFlagOut + i * blocksPerStream, numOfBlock * sizeof(CompressFlagBlock),
            cudaMemcpyDeviceToHost, cudaStreams[i]),
            "Failed to copy deviceFlagOut to host");
    }

    cudaCheckError(cudaMemcpy(hostOutSize, deviceOutSize, numOfStreams * sizeof(int), cudaMemcpyDeviceToHost),
        "Failed to copy deviceOutSize to host");
    cudaCheckError(cudaMemcpy(hostFlagSize, deviceFlagSize, numOfStreams * sizeof(int), cudaMemcpyDeviceToHost),
        "Failed to copy deviceFlagSize to host");
    cudaDeviceSynchronize();
    printf("%.6fs\n", timer.end());
    fflush(stdout);

    // Post process ------------------------------
    outSize = flagSize = 0;
    for (int i = 0; i < numOfStreams; ++i) {
        outSize += hostOutSize[i];
        flagSize += hostFlagSize[i];
    }

    // Cleanup -----------------------------------
    for (int i = 0; i < numOfStreams; ++i) {
        cudaCheckError(cudaStreamDestroy(cudaStreams[i]), "Failed to destroy CUDA streams");
    }

    delete[] cudaStreams;
    delete[] hostOutSize;
    delete[] hostFlagSize;

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

    uint8_t *deviceInBuf, *deviceOutBuf;
    CompressFlagBlock* deviceFlagIn;

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
    printf("Launching kernel... ");
    fflush(stdout);

    timer.begin();
    auto dimGrid = (nFlagBlocks - 1) / GPUBlockSize + 1;
    DecompressKernel<<<dimGrid, GPUBlockSize>>>(deviceFlagIn, nFlagBlocks, deviceInBuf, deviceOutBuf);
    cudaCheckError(cudaDeviceSynchronize(), "Failed to launch kernel");
    auto elapsed = timer.end();
    printf("%.6fs\n", elapsed);

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
