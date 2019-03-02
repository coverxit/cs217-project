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

    uint8_t **deviceInBuf, **deviceOutBuf;
    CompressFlagBlock** deviceFlagOut;
    int **deviceOutSize, **deviceFlagSize;
    int *hostOutSize, *hostFlagSize;

    int numOfGPUs = 0;
    cudaGetDeviceCount(&numOfGPUs);

    auto numOfKernels = std::min(nFlagBlocks / numOfGPUs, numOfGPUs);
    auto blocksPerKernel = (nFlagBlocks - 1) / numOfKernels + 1;
    auto alignedKernelSize = blocksPerKernel * DataBlockSize;

    // Allocate ----------------------------------
    printf("Allocating device variables... ");
    timer.begin();

    deviceInBuf = new uint8_t*[numOfKernels];
    deviceOutBuf = new uint8_t*[numOfKernels];
    deviceFlagOut = new CompressFlagBlock*[numOfKernels];
    deviceOutSize = new int*[numOfKernels];
    deviceFlagSize = new int*[numOfKernels];

    hostOutSize = new int[numOfKernels];
    hostFlagSize = new int[numOfKernels];

    for (int i = 0; i < numOfKernels; ++i) {
        cudaCheckError(cudaSetDevice(i), "Failed to set device");

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

    // Copy: host to device -----------------------
    printf("Copying from host to device => GPU");
    timer.begin();

    for (int i = 0; i < numOfKernels; ++i) {
        cudaCheckError(cudaSetDevice(i), "Failed to set device");

        auto kernelSize = std::min(alignedKernelSize, inSize - i * alignedKernelSize);
        auto numOfBlock = std::min(blocksPerKernel, nFlagBlocks - i * blocksPerKernel);

        printf(" [%d]", i);
        fflush(stdout);
        
        cudaCheckError(cudaMemcpyAsync(deviceInBuf[i],
                           inBuf + i * alignedKernelSize, kernelSize,
                           cudaMemcpyHostToDevice),
            "Failed to copy inBuf to device");

        cudaCheckError(cudaMemsetAsync(deviceFlagOut[i], 0, sizeof(CompressFlagBlock) * numOfBlock),
            "Failed to set deviceFlagOut to 0");

        cudaCheckError(cudaMemsetAsync(deviceOutSize[i], 0, sizeof(int)), "Failed to set deviceOutSize to 0");
        cudaCheckError(cudaMemsetAsync(deviceFlagSize[i], 0, sizeof(int)), "Failed to set deviceFlagSize to 0");
    }

    for (int i = 0; i < numOfKernels; ++i) {
        cudaCheckError(cudaSetDevice(i), "Failed to set device");
        cudaCheckError(cudaDeviceSynchronize(), "Failed to synchronize");
    }
    printf("... %.6fs\n", timer.end());
    
    // Launch kernel ----------------------------------
    printf("Launching kernel => GPU");
    timer.begin();

    for (int i = 0; i < numOfKernels; ++i) {
        cudaCheckError(cudaSetDevice(i), "Failed to set device");

        auto kernelSize = std::min(alignedKernelSize, inSize - i * alignedKernelSize);
        auto numOfBlock = std::min(blocksPerKernel, nFlagBlocks - i * blocksPerKernel);

        printf(" [%d]", i);
        fflush(stdout);

        CompressKernel<<<numOfBlock, GPUBlockSize>>>(
            deviceInBuf[i], kernelSize,
            deviceOutBuf[i], deviceOutSize[i],
            deviceFlagOut[i], deviceFlagSize[i]);
    }

    for (int i = 0; i < numOfKernels; ++i) {
        cudaCheckError(cudaSetDevice(i), "Failed to set device");
        cudaCheckError(cudaDeviceSynchronize(), "Failed to launch multi-GPU kernel");
    }
    auto elasped = timer.end();
    printf("... %.6fs\n", elasped);

    // Copy: device to host ---------------------------
    printf("Copying from device to host => GPU");
    timer.begin();

    for (int i = 0; i < numOfKernels; ++i) {
        cudaCheckError(cudaSetDevice(i), "Failed to set device");

        auto kernelSize = std::min(alignedKernelSize, inSize - i * alignedKernelSize);
        auto numOfBlock = std::min(blocksPerKernel, nFlagBlocks - i * blocksPerKernel);

        printf(" [%d]", i);
        fflush(stdout);

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

    for (int i = 0; i < numOfKernels; ++i) {
        cudaCheckError(cudaSetDevice(i), "Failed to set device");
        cudaCheckError(cudaDeviceSynchronize(), "Failed to synchronize");
    }
    printf("... %.6fs\n", timer.end());

    printf("Post processing and cleanup... ");
    timer.begin();

    // Post process ------------------------------
    outSize = flagSize = 0;
    for (int i = 0; i < numOfKernels; ++i) {
        outSize += hostOutSize[i];
        flagSize += hostFlagSize[i];
    }

    for (int i = 0; i < numOfKernels; ++i) {
        cudaCheckError(cudaSetDevice(i), "Failed to set device");
        cudaFree(deviceInBuf[i]);
        cudaFree(deviceOutBuf[i]);
        cudaFree(deviceOutSize[i]);
        cudaFree(deviceFlagOut[i]);
        cudaFree(deviceFlagSize[i]);
    }

    delete[] deviceInBuf;
    delete[] deviceOutBuf;
    delete[] deviceFlagOut;
    delete[] deviceOutSize;
    delete[] deviceFlagSize;

    delete[] hostOutSize;
    delete[] hostFlagSize;

    printf("%.6fs\n", timer.end());
    return std::make_pair(true, elasped);
}

std::pair<bool, double> CUDALZSS::decompress(CompressFlagBlock* flagIn, int nFlagBlocks,
    const uint8_t* inBuf, int inSize, uint8_t* outBuf, int outSize)
{
    Timer timer(false);

    uint8_t **deviceInBuf, **deviceOutBuf;
    CompressFlagBlock** deviceFlagIn;

    int numOfGPUs = 0;
    cudaGetDeviceCount(&numOfGPUs);

    auto totalGPUBlocks = (nFlagBlocks - 1) / GPUBlockSize + 1;
    auto numOfKernels = std::min(totalGPUBlocks / numOfGPUs, numOfGPUs);
    auto gpuBlocksPerKernel = (totalGPUBlocks - 1) / numOfKernels + 1;
    auto dataBlocksPerKernel = gpuBlocksPerKernel * GPUBlockSize;
    auto alignedKernelSize = dataBlocksPerKernel * DataBlockSize;
    
    // Allocate ----------------------------------
    printf("Allocating device variables... ");
    timer.begin();

    deviceInBuf = new uint8_t*[numOfKernels];
    deviceOutBuf = new uint8_t*[numOfKernels];
    deviceFlagIn = new CompressFlagBlock*[numOfKernels];

    for (int i = 0; i < numOfKernels; ++i) {
        cudaCheckError(cudaSetDevice(i), "Failed to set device");

        auto kernelOutSize = std::min(alignedKernelSize, outSize - i * alignedKernelSize);
        auto numOfDataBlock = std::min(dataBlocksPerKernel, nFlagBlocks - i * dataBlocksPerKernel);

        cudaCheckError(cudaMalloc((void**)&deviceInBuf[i], inSize), "Failed to allocate deviceInBuf");
        cudaCheckError(cudaMalloc((void**)&deviceOutBuf[i], kernelOutSize), "Failed to allocate deviceOutBuf");
        cudaCheckError(cudaMalloc((void**)&deviceFlagIn[i], sizeof(CompressFlagBlock) * numOfDataBlock),
            "Failed to allocate deviceFlagIn");

        cudaDeviceSynchronize();
    }
    printf("%.6fs\n", timer.end());

    // Copy: host to device -----------------------
    printf("Copying from host to device => GPU");
    timer.begin();

    for (int i = 0; i < numOfKernels; ++i) {
        cudaCheckError(cudaSetDevice(i), "Failed to set device");

        auto numOfDataBlock = std::min(dataBlocksPerKernel, nFlagBlocks - i * dataBlocksPerKernel);

        printf(" [%d]", i);
        fflush(stdout);

        cudaCheckError(cudaMemcpyAsync(deviceInBuf[i], inBuf, inSize, cudaMemcpyHostToDevice),
            "Failed to copy inBuf to device");

        cudaCheckError(cudaMemcpyAsync(deviceFlagIn[i],
                           flagIn + i * dataBlocksPerKernel, sizeof(CompressFlagBlock) * numOfDataBlock,
                           cudaMemcpyHostToDevice),
            "Failed to copy flagIn to device");
    }

    for (int i = 0; i < numOfKernels; ++i) {
        cudaCheckError(cudaSetDevice(i), "Failed to set device");
        cudaCheckError(cudaDeviceSynchronize(), "Failed to synchronize");
    }
    printf("... %.6fs\n", timer.end());

    // Launch kernel ----------------------------------
    printf("Launching kernel => GPU ");
    timer.begin();

    for (int i = 0; i < numOfKernels; ++i) {
        cudaCheckError(cudaSetDevice(i), "Failed to set device");

        auto kernelInSize = std::min(alignedKernelSize, inSize - i * alignedKernelSize);
        auto numOfDataBlock = std::min(dataBlocksPerKernel, nFlagBlocks - i * dataBlocksPerKernel);
        auto numOfGPUBlock = std::min(gpuBlocksPerKernel, totalGPUBlocks - i * gpuBlocksPerKernel);

        printf(" [%d]", i);
        fflush(stdout);

        DecompressKernel<<<numOfGPUBlock, GPUBlockSize>>>(deviceFlagIn[i], numOfDataBlock, 
            deviceInBuf[i], deviceOutBuf[i]);
    }

    for (int i = 0; i < numOfKernels; ++i) {
        cudaCheckError(cudaSetDevice(i), "Failed to set device");
        cudaCheckError(cudaDeviceSynchronize(), "Failed to launch multi-GPU kernel");
    }
    auto elasped = timer.end();
    printf("... %.6fs\n", elasped);

    // Copy: device to host ---------------------------
    printf("Copying from device to host => GPU");
    timer.begin();

    for (int i = 0; i < numOfKernels; ++i) {
        cudaCheckError(cudaSetDevice(i), "Failed to set device");

        auto kernelOutSize = std::min(alignedKernelSize, outSize - i * alignedKernelSize);

        printf(" [%d]", i);
        fflush(stdout);

        cudaCheckError(cudaMemcpyAsync(outBuf + i * alignedKernelSize,
                           deviceOutBuf[i], kernelOutSize,
                           cudaMemcpyDeviceToHost),
            "Failed to copy deviceOutBuf to host");
    }

    for (int i = 0; i < numOfKernels; ++i) {
        cudaCheckError(cudaSetDevice(i), "Failed to set device");
        cudaCheckError(cudaDeviceSynchronize(), "Failed to synchronize");
    }
    printf("... %.6fs\n", timer.end());

    for (int i = 0; i < numOfKernels; ++i) {
        cudaCheckError(cudaSetDevice(i), "Failed to set device");
        cudaFree(deviceInBuf[i]);
        cudaFree(deviceOutBuf[i]);
        cudaFree(deviceFlagIn[i]);
    }

    delete[] deviceInBuf;
    delete[] deviceOutBuf;
    delete[] deviceFlagIn;

    return std::make_pair(true, -1);
}
