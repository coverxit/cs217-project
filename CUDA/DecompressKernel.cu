#include <utility>

#include <assert.h>
#include <memory.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#include "../Settings.h"

#include "../BitHelper.h"
#include "../LZSSInterface.h"

// Minor difference to the implementation of BlockDecompress.
__global__ void DecompressKernel(CompressFlagBlock* deviceFlagIn,
    const uint8_t* deviceInBuf, uint8_t* deviceOutBuf)
{
    __shared__ uint8_t blockFlags[DataBlockSize / 8];
    // Also used to store literal (in high byte, low byte is set as 0)
    __shared__ PairType replacePairs[DataBlockSize];
    __shared__ uint16_t writtenSize[DataBlockSize];

    auto blockId = blockIdx.x;
    auto threadId = threadIdx.x;

    auto compressedOffset = deviceFlagIn[blockId].CompressedOffset;
    auto compressedSize = deviceFlagIn[blockId].CompressedSize;
    auto numOfFlags = deviceFlagIn[blockId].NumOfFlags;

    for (int t = threadId; t < SIZE_OF_FLAGS(numOfFlags); t += blockDim.x) {
        blockFlags[t] = deviceFlagIn[blockId].Flags[t];

        // 8 bits per flag
        for (int i = 0; i < 8; ++i) {
            if (t * 8 + i < numOfFlags) {
                // +1 for its taken size in inBuf
                replacePairs[t * 8 + i] = ((blockFlags[t] >> i) & 0x1) + 1;
            }
        }
    }
    
    // Prefix sum for calculating reading size in inBuf
    for (int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();

        if (threadId >= stride) {
            for (int t = threadId; t < numOfFlags; t += blockDim.x) {
                replacePairs[t] += replacePairs[t - stride];
            }  
        }
    }
    __syncthreads();

    // Fetch all pairs in inBuf & prepare writtenSize
    for (int t = threadId; t < numOfFlags; t += blockDim.x) {
        if (GET_BIT(blockFlags, t) == 0) {
            replacePairs[t] = deviceInBuf[compressedOffset + replacePairs[t] - 1] << 8;
            writtenSize[t] = 1;
        } else {
            replacePairs[t] = (deviceInBuf[compressedOffset + replacePairs[t] - 1] << 8)
                | (deviceInBuf[compressedOffset + replacePairs[t] - 2]);
            writtenSize[t] = (replacePairs[t] & (MaxEncodeLength - 1)) + 1;
        }
    }
    
    // Prefix sum for calculating written size in outBuf
    for (int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();

        if (threadId >= stride) {
            for (int t = threadId; t < numOfFlags; t += blockDim.x) {
                writtenSize[t] += writtenSize[t - stride];
            }  
        }
    }
    __syncthreads();

    /*
    auto inOffset = 0;
    auto outOffset = 0;

    for (int j = 0; j < flagBlock.NumOfFlags; ++j) {
        if (GET_BIT(flagBlock.Flags, j) == 0) {
            // Single character
            outBuf[outOffset] = inBuf[inOffset];
            ++inOffset;
            ++outOffset;
        } else {
            // Replacement pair
            PairType matchPair;
            memcpy(&matchPair, inBuf + inOffset, sizeof(PairType));

            // Plus 1 for the opposite operation in compression
            auto matchOffset = (matchPair >> PairLengthBits) + 1;
            auto matchLength = (matchPair & (MaxEncodeLength - 1)) + 1;

            // May overlap, so manually copy
            for (int k = 0; k < matchLength; ++k) {
                outBuf[outOffset] = outBuf[outOffset - matchOffset];
                ++outOffset;
            }

            inOffset += 2;
        }
        }

        // Copy back to global memory
        memcpy(deviceOutBuf + blockId * DataBlockSize, outBuf, outOffset);
    */
}
