#include <memory.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#include "../Settings.h"

#include "../BitHelper.h"
#include "../TimerHelper.hpp"

#include "../MatchHelper/MatchHelper.h"

__global__ void CompressKernel(const uint8_t* deviceInBuf, int inSize,
    uint8_t* deviceOutBuf, int* deviceOutSize,
    CompressFlagBlock* deviceFlagOut, int nFlagBlocks, int* deviceFlagSize,
    int* deviceNumBlocksDone)
{
    __shared__ uint8_t blockBuf[DataBlockSize];
    __shared__ PairType blockFlags[DataBlockSize];

    auto threadId = threadIdx.x;
    auto blockId = blockIdx.x;

    auto blockOffset = blockId * DataBlockSize;
    auto blockSize = MIN(DataBlockSize, inSize - blockOffset);

    for (int t = threadId; t < blockSize; t += blockDim.x) {
        blockBuf[t] = deviceInBuf[blockOffset + t];
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
        CompressFlagBlock compressBlock;
        memset(&compressBlock, 0, sizeof(CompressFlagBlock));

        for (int i = 0; i < blockSize;) {
            if (blockFlags[i] == 0) {
                deviceOutBuf[blockOffset + compressBlock.CompressedSize] = blockBuf[i];
                ++compressBlock.CompressedSize;

                PUT_BIT(compressBlock.Flags, compressBlock.NumOfFlags, 0);
                i += 1;
            } else {
                memcpy(deviceOutBuf + blockOffset + compressBlock.CompressedSize, &blockFlags[i], sizeof(PairType));
                compressBlock.CompressedSize += sizeof(PairType);

                PUT_BIT(compressBlock.Flags, compressBlock.NumOfFlags, 1);

                auto matchLength = (blockFlags[i] & (MaxEncodeLength - 1)) + 1;
                i += matchLength;
            }

            ++compressBlock.NumOfFlags;
        }

        memcpy(deviceFlagOut + blockId, &compressBlock, sizeof(CompressFlagBlock));

        // taken by current flag block
        atomicAdd(deviceFlagSize, SIZE_OF_FLAGS(compressBlock.NumOfFlags) 
            + sizeof(CompressFlagBlock::NumOfFlags) + sizeof(CompressFlagBlock::CompressedSize));
        atomicAdd(deviceOutSize, compressBlock.CompressedSize);

        // Fetch and add
        auto numBlocksDone = atomicAdd(deviceNumBlocksDone, 1) + 1;
        if (numBlocksDone % 100 == 0) {
            printf("Block %d/%d done.\n", numBlocksDone, nFlagBlocks);
        }
    }
}
