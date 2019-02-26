#include <utility>

#include <memory.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#include "../Settings.h"

#include "../BitHelper.h"
#include "../LZSSInterface.h"

// Minor difference to the implementation of BlockDecompress.
__global__ void DecompressKernel(CompressFlagBlock* deviceFlagIn, int nFlagBlocks,
    const uint8_t* deviceInBuf, uint8_t* deviceOutBuf)
{
    CompressFlagBlock flagBlock;
    uint8_t inBuf[DataBlockSize], outBuf[DataBlockSize];

    // This is the data blockId. Each thread is responsible for a 4KB data block.
    auto blockId = blockIdx.x * blockDim.x + threadIdx.x;

    if (blockId < nFlagBlocks) {
        memcpy(&flagBlock, deviceFlagIn + blockId, sizeof(CompressFlagBlock));
        memcpy(inBuf, deviceInBuf + flagBlock.CompressedOffset, flagBlock.CompressedSize);

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
    }
}
