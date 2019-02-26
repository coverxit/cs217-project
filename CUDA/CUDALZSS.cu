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

#define cudaCheckError(op, msg) \
    do { cudaError_t ret = (op); if ((ret) != cudaSuccess) _gerror((ret), (msg)); } while (false)

#define MIN(a, b) \
    ((a) < (b) ? (a) : (b))

inline void _gerror(cudaError_t cudaError, const char* msg) {
    fprintf(stderr, "%s, cudaError = %d\n", msg, cudaError);
    exit(-1);
}

__global__ void CompressKernel(const uint8_t* deviceInBuf, int inSize,
    uint8_t* deviceOutBuf, int* deviceOutSize,
    CompressFlagBlock* deviceFlagOut, int nFlagBlocks, int* deviceFlagSize)
{
    __shared__ uint8_t blockBuf[DataBlockSize];
    __shared__ PairType blockFlags[DataBlockSize];
    __shared__ CompressFlagBlock compressBlock;

    auto threadId = threadIdx.x;
    auto blockId = blockIdx.x;

    auto blockOffset = blockIdx.x * DataBlockSize;
    auto blockSize = MIN(DataBlockSize, inSize - blockOffset);

    for (int t = threadId; t < blockSize; t += blockDim.x) {
        blockBuf[t] = deviceInBuf[blockOffset + t];
    }

    for (int t = threadId; t < DataBlockSize / 8; t += blockDim.x) {
        compressBlock.Flags[t] = 0;
    }
    __syncthreads();

    for (int t = threadId; t < blockSize; t += blockDim.x) {
        auto lookbackLength = MIN(WindowSize, t);
        auto lookaheadLength = MIN(MaxEncodeLength, blockSize - t);
        int matchOffset, matchLength;

        if (FindMatch(blockBuf + t - lookbackLength, lookbackLength,
            blockBuf + t, lookaheadLength, matchOffset, matchLength)) {

            // Convert offset to backward representation
            matchOffset = lookbackLength - matchOffset;

            // Due to the bit limit, minus 1 for exact offset and length
            blockFlags[t] = ((matchOffset - 1) << PairLengthBits) | (matchLength - 1);
        } else {
            blockFlags[t] = 0;
        }
    }
    __syncthreads();

    // Collector
    if (threadId == 0) {
        compressBlock.CompressedSize = 0;
        compressBlock.NumOfFlags = 0;
        
        for (int i = 0; i < blockSize; ) {
            if (blockFlags[i] == 0) {
                deviceOutBuf[blockOffset + compressBlock.CompressedSize] = blockBuf[i];
                ++compressBlock.CompressedSize;

                PUT_BIT(compressBlock.Flags, compressBlock.NumOfFlags, 0);
                i += 1;
            } else {
                // Plus 1 for the opposite operation in compression
                auto matchLength = (blockFlags[i] & (MaxEncodeLength - 1)) + 1;

                memcpy(deviceOutBuf + blockOffset + compressBlock.CompressedSize, &blockFlags[i], sizeof(PairType));
                compressBlock.CompressedSize += sizeof(PairType);

                PUT_BIT(compressBlock.Flags, compressBlock.NumOfFlags, 1);
                i += matchLength;
            }

            ++compressBlock.NumOfFlags;
        }

        memcpy(deviceFlagOut + blockId, &compressBlock, sizeof(CompressFlagBlock));

        // taken by current flag block
        atomicAdd(deviceFlagSize, SIZE_OF_FLAGS(compressBlock.NumOfFlags) + sizeof(CompressFlagBlock::NumOfFlags)
            + sizeof(CompressFlagBlock::CompressedSize));
        atomicAdd(deviceOutSize, compressBlock.CompressedSize);

        printf("Block %d/%d done.\n", blockId, nFlagBlocks);
    }
}

std::pair<bool, double> CUDALZSS::compress(const uint8_t* inBuf, int inSize,
    uint8_t* outBuf, int& outSize,
    CompressFlagBlock* flagOut, int nFlagBlocks, int& flagSize)
{
    Timer timer(false);

    uint8_t *deviceInBuf, *deviceOutBuf;
    CompressFlagBlock *deviceFlagOut;
    int *deviceOutSize, *deviceFlagSize;

    // Allocate ----------------------------------
    printf("Allocating device variables...\n");
    cudaCheckError(cudaMalloc((void**) &deviceInBuf, inSize), "Failed to allocate deviceInBuf");

    cudaCheckError(cudaMalloc((void**) &deviceOutBuf, outSize), "Failed to allocate deviceOutBuf");
    cudaCheckError(cudaMalloc((void**) &deviceOutSize, sizeof(int)), "Failed to allocate deviceOutSize");

    cudaCheckError(cudaMalloc((void**) &deviceFlagOut, sizeof(CompressFlagBlock) * nFlagBlocks), 
        "Failed to allocate deviceFlagOut");
    cudaCheckError(cudaMalloc((void**) &deviceFlagSize, sizeof(int)), "Failed to allocate deviceFlagSize");

    // Copy: host to device -----------------------
    printf("Copying data from host to device...\n");
    cudaCheckError(cudaMemcpy(deviceInBuf, inBuf, inSize, cudaMemcpyHostToDevice), 
        "Failed to copy deviceInBuf to device");
    cudaCheckError(cudaMemset(deviceFlagOut, 0, sizeof(CompressFlagBlock) * nFlagBlocks), 
        "Failed to set deviceFlagOut to 0");
    cudaDeviceSynchronize();

    // Launch kernel ------------------------------
    printf("Launching kernel...\n");

    timer.begin();
    CompressKernel<<<nFlagBlocks, GPUBlockSize>>>(deviceInBuf, inSize, 
        deviceOutBuf, deviceOutSize, deviceFlagOut, nFlagBlocks, deviceFlagSize);
    auto elapsed = timer.end();
    cudaCheckError(cudaDeviceSynchronize(), "Failed to launch kernel");

    // Copy: device to host -----------------------
    printf("Copying data from device to host...\n");
    cudaCheckError(cudaMemcpy(&outSize, deviceOutSize, sizeof(int), cudaMemcpyDeviceToHost),
        "Failed to copy deviceOutSize to host");
    cudaCheckError(cudaMemcpy(&flagSize, deviceFlagSize, sizeof(int), cudaMemcpyDeviceToHost),
        "Failed to copy deviceFlagSize to host");

    cudaCheckError(cudaMemcpy(outBuf, deviceOutBuf, outSize, cudaMemcpyDeviceToHost), 
        "Failed to copy deviceOutBuf to host");
    cudaCheckError(cudaMemcpy(flagOut, deviceFlagOut, sizeof(CompressFlagBlock) * nFlagBlocks, cudaMemcpyDeviceToHost),
        "Failed to copy deviceFlagOut to host");
    cudaDeviceSynchronize();
    
    cudaFree(deviceInBuf);
    cudaFree(deviceOutBuf);
    cudaFree(deviceOutSize);
    cudaFree(deviceFlagOut);
    cudaFree(deviceFlagSize);

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
