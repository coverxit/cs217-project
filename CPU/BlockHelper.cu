#include <algorithm>
#include <functional>

#include <memory.h>
#include <stdint.h>

#include "../Settings.h"

#include "../BitHelper.h"
#include "../LZSSInterface.h"
#include "../MatchHelper/MatchHelper.h"

#include "BlockHelper.h"

void blockCompress(int blockId,
    const uint8_t* inBuf, int inSize,
    uint8_t* outBuf, int& outSize,
    CompressFlagBlock* flagOut, int& flagSize,
    std::function<void(int)> finishCallback)
{
    uint8_t blockBuf[DataBlockSize];
    auto blockOffset = blockId * DataBlockSize;
    auto blockSize = std::min(DataBlockSize, inSize - blockOffset);

    memcpy(blockBuf, inBuf + blockOffset, blockSize);

    // The first character
    auto nFlags = 1;
    auto written = 1;

    PUT_BIT(flagOut[blockId].Flags, 0, 0);
    outBuf[blockOffset] = blockBuf[0];

    // Later on
    for (int j = 1; j < blockSize;) {
        auto lookbackLength = std::min(WindowSize, j);
        auto lookaheadLength = std::min(MaxEncodeLength, blockSize - j);
        int matchOffset, matchLength;

        if (FindMatch(blockBuf + j - lookbackLength, lookbackLength,
                blockBuf + j, lookaheadLength,
                matchOffset, matchLength)) {

            // Convert offset to backward representation
            matchOffset = lookbackLength - matchOffset;

            // Due to the bit limit, minus 1 for exact offset and length
            PairType matchPair = ((matchOffset - 1) << PairLengthBits) | (matchLength - 1);
            memcpy(outBuf + blockOffset + written, &matchPair, sizeof(PairType));
            written += sizeof(PairType);

            PUT_BIT(flagOut[blockId].Flags, nFlags, 1);
            j += matchLength;
        } else {
            outBuf[blockOffset + written] = blockBuf[j];
            written += 1;

            PUT_BIT(flagOut[blockId].Flags, nFlags, 0);
            j += 1;
        }

        ++nFlags;
    }

    flagOut[blockId].NumOfFlags = (uint16_t)nFlags;
    flagOut[blockId].CompressedSize = written;

    // taken by current flag block
    flagSize += SIZE_OF_FLAGS(nFlags) + sizeof(CompressFlagBlock::NumOfFlags)
        + sizeof(CompressFlagBlock::CompressedSize);
    outSize += written;

    if (finishCallback != nullptr) {
        finishCallback(blockId);
    }
}

void blockDecompress(int blockId,
    CompressFlagBlock* flagIn, int nFlagBlocks,
    const uint8_t* inBuf, int inSize, uint8_t* outBuf,
    std::function<void(int)> finishCallback)
{
    auto inOffset = flagIn[blockId].CompressedOffset;
    auto outOffset = blockId * DataBlockSize;

    for (int j = 0; j < flagIn[blockId].NumOfFlags; ++j) {
        if (GET_BIT(flagIn[blockId].Flags, j) == 0) {
            // Single character
            outBuf[outOffset] = inBuf[inOffset];
            ++inOffset;
            ++outOffset;
        } else {
            // Replacement pair
            PairType matchPair = *(uint16_t*)&inBuf[inOffset];

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

    if (finishCallback != nullptr) {
        finishCallback(blockId);
    }
}