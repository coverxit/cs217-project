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
    Timer timer(false), timerKernel;

    uint8_t **deviceInBuf, **deviceOutBuf;
    CompressFlagBlock** deviceFlagOut;
    int **deviceOutSize, **deviceFlagSize;
    int *hostOutSize, *hostFlagSize;

    int numOfGPUs = 0;
    cudaGetDeviceCount(&numOfGPUs);

    deviceOutSize = new int*[numOfKernels];
    deviceFlagSize = new int*[numOfKernels];
    hostOutSize = new int[numOfKernels];
    hostFlagSize = new int[numOfKernels];

    auto numOfKernels = std::min(nFlagBlocks / numOfGPUs, numOfGPUs);
    auto blocksPerKernel = (nFlagBlocks - 1) / numOfKernels + 1;
    auto alignedKernelSize = blocksPerKernel * DataBlockSize;

    // Allocate ----------------------------------
    printf("Allocating device variables... ");
    timer.begin();

    for (int i = 0; i < numOfKernels; ++i) {
        cudaSetDevice(i);

        auto kernelSize = std::min(alignedKernelSize, inSize - i * alignedKernelSize);
        auto numOfBlock = std::min(blocksPerKernel, nFlagBlocks - i * blocksPerKernel);
        
        cudaCheckError(cudaMalloc((void**)&deviceInBuf[i], kernelSize), "Failed to allocate deviceInBuf");

        cudaCheckError(cudaMalloc((void**)&deviceOutBuf[i], kernelSize), "Failed to allocate deviceOutBuf");
        cudaCheckError(cudaMalloc((void**)&deviceOutSize[i], sizeof(int)), "Failed to allocate deviceOutSize");

        cudaCheckError(cudaMalloc((void**)&deviceFlagOut[i], sizeof(CompressFlagBlock) * numOfBlock),
            "Failed to allocate deviceFlagOut");
        cudaCheckError(cudaMalloc((void**)&deviceFlagSize[i], sizeof(int)), "Failed to allocate deviceFlagSize");

        cudaDeviceSynchronize();
    }
    printf("%.6fs\n", timer.end());

    // Multi-GPU ----------------------------------
    printf("Launching kernels on multiple GPUs...\n  => GPU:");
    timer.begin();
    
    for (int i = 0; i < numOfKernels; ++i) {
        cudaSetDevice(i);

        auto kernelSize = std::min(alignedKernelSize, inSize - i * alignedKernelSize);
        auto numOfBlock = std::min(blocksPerKernel, nFlagBlocks - i * blocksPerKernel);

        // Copy: host to device -------------------
        cudaCheckError(cudaMemcpyAsync(deviceInBuf[i], 
            inBuf + i * alignedKernelSize, kernelSize, 
            cudaMemcpyHostToDevice),
            "Failed to copy inBuf to device");

        cudaCheckError(cudaMemsetAsync(deviceFlagOut[i], 0, sizeof(CompressFlagBlock) * numOfBlock),
            "Failed to set deviceFlagOut to 0");

        cudaCheckError(cudaMemsetAsync(deviceOutSize[i], 0, sizeof(int)),  "Failed to set deviceOutSize to 0");
        cudaCheckError(cudaMemsetAsync(deviceFlagSize[i], 0, sizeof(int)),  "Failed to set deviceFlagSize to 0");

        printf(" [%d]", i);
        fflush(stdout);

        // Launch kernel ------------------------------
        CompressKernel<<<numOfBlock, GPUBlockSize>>>(
            deviceInBuf[i], kernelSize,
            deviceOutBuf[i], deviceOutSize[i],
            deviceFlagOut[i], deviceFlagSize[i]);

        // Copy: device to host -----------------------
        cudaCheckError(cudaMemcpyAsync(outBuf + i * alignedKernelSize, 
            deviceOutBuf[i], kernelSize, 
            cudaMemcpyDeviceToHost),
            "Failed to copy deviceOutBuf to host");

        cudaCheckError(cudaMemcpyAsync(flagOut + i * blocksPerKernel,
            deviceFlagOut[i], numOfBlock * sizeof(CompressFlagBlock),
            cudaMemcpyDeviceToHost),
            "Failed to copy deviceFlagOut to host");
        
        cudaCheckError(cudaMemcpyAsync(hostOutSize + i, deviceOutSize[i], 
            sizeof(int), cudaMemcpyDeviceToHost),
            "Failed to copy deviceOutSize to host");
        cudaCheckError(cudaMemcpyAsync(hostFlagSize + i, deviceFlagSize[i], 
            sizeof(int), cudaMemcpyDeviceToHost),
            "Failed to copy deviceFlagSize to host");

    }

    printf("\nWaiting for kernel exeuction complete... ");
    for (int i = 0; i < numOfKernels; ++i) {
        cudaSetDevice(i);
        cudaCheckError(cudaDeviceSynchronize(), "Failed to launch multi-GPU kernel");
    }
    printf("%.6fs\n", timer.end());

    printf("Post processing and cleanup... ");
    timer.begin();

    // Post process ------------------------------
    outSize = flagSize = 0;
    for (int i = 0; i < numOfKernels; ++i) {
        outSize += hostOutSize[i];
        flagSize += hostFlagSize[i];
    }

    for (int i = 0; i < numOfKernels; ++i) {
        cudaSetDevice(i);

        cudaFree(deviceInBuf[i]);
        cudaFree(deviceOutBuf[i]);
        cudaFree(deviceOutSize[i]);
        cudaFree(deviceFlagOut[i]);
        cudaFree(deviceFlagSize[i]);
    }

    delete[] deviceOutSize;
    delete[] deviceFlagSize;
    delete[] hostOutSize;
    delete[] hostFlagSize;

    printf("%.6fs\n", timer.end());
    return std::make_pair(true, timerKernel.end());
}

std::pair<bool, double> CUDALZSS::decompress(CompressFlagBlock* flagIn, int nFlagBlocks,
    const uint8_t* inBuf, int inSize, uint8_t* outBuf, int outSize)
{
    Timer timer(false), timerKernel;

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
    printf("%.6fs\n", timer.end());

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

    return std::make_pair(true, timerKernel.end());
}
