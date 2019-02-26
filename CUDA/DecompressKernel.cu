#include <utility>

#include <memory.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#include "../Settings.h"

#include "../BitHelper.h"
#include "../LZSSInterface.h"

// Minor difference to the implementation of BlockDecompress.
__global__ void DecompressKernel(const uint8_t* deviceFlagIn, int nFlagBlocks,
    const uint8_t* deviceInBuf, uint8_t* deviceOutBuf)
{
    CompressFlagBlock flagBlock;

    // This is the data blockId. Each thread is responsible for a 4KB data block.
    auto blockId = blockIdx.x * blockDim.x + threadIdx.x;

    // Manually byte-by-byte copy, otherwise misaligned memory access is triggered.
    auto copyOffestLocal = (uint8_t*)&flagBlock;
    auto copyOffsetGlobal = deviceFlagIn + blockId * sizeof(CompressFlagBlock);
    for (int i = 0; i < sizeof(CompressFlagBlock); ++i) {
        *(copyOffestLocal++) = *(copyOffsetGlobal++);
    }

    if (blockId < nFlagBlocks) {
        auto inOffset = flagBlock.CompressedOffset;
        auto outOffset = blockId * DataBlockSize;

        for (int j = 0; j < flagBlock.NumOfFlags; ++j) {
            if (GET_BIT(flagBlock.Flags, j) == 0) {
                // Single character
                deviceOutBuf[outOffset] = deviceInBuf[inOffset];
                ++inOffset;
                ++outOffset;
            } else {
                // Replacement pair
                PairType matchPair = *(uint16_t*)&deviceInBuf[inOffset];

                // Plus 1 for the opposite operation in compression
                auto matchOffset = (matchPair >> PairLengthBits) + 1;
                auto matchLength = (matchPair & (MaxEncodeLength - 1)) + 1;

                // May overlap, so manually copy
                for (int k = 0; k < matchLength; ++k) {
                    deviceOutBuf[outOffset] = deviceOutBuf[outOffset - matchOffset];
                    ++outOffset;
                }

                inOffset += 2;
            }
        }
    }
}
