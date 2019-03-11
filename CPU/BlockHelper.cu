#include <algorithm>
#include <functional>

#include <memory.h>
#include <stdint.h>

#include "../Settings.h"

#include "../BitHelper.h"
#include "../LZSSInterface.h"
#include "../MatchHelper/MatchHelper.h"

#include "BlockHelper.h"

void BlockCompress(int64_t blockId,
    const uint8_t* inBuf, int64_t inSize,
    uint8_t* outBuf, int64_t& outSize,
    CompressFlagBlock* flagOut, int64_t& flagSize,
    std::function<void(int64_t)> finishCallback)
{
    int64_t blockOffset = blockId * DataBlockSize;
    int64_t blockSize = std::min(DataBlockSize, inSize - blockOffset);
    int64_t blockBuf = inBuf + blockOffset;

    int64_t nFlags = 0, written = 0;
    for (int j = 0; j < blockSize;) {
        auto lookbackLength = std::min(WindowSize, j);
        auto lookaheadLength = std::min(MaxEncodeLength, blockSize - j);
        int matchOffset, matchLength;

        if (FindMatch(blockBuf + j - lookbackLength, lookbackLength,
                blockBuf + j, lookaheadLength, matchOffset, matchLength)) {

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
            ++written;

            PUT_BIT(flagOut[blockId].Flags, nFlags, 0);
            ++j;
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

void BlockDecompress(int64_t blockId,
    CompressFlagBlock* flagIn,
    const uint8_t* inBuf, uint8_t* outBuf,
    std::function<void(int64_t)> finishCallback)
{
    int64_t inOffset = flagIn[blockId].CompressedOffset;
    int64_t outOffset = blockId * DataBlockSize;

    for (int j = 0; j < flagIn[blockId].NumOfFlags; ++j) {
        if (GET_BIT(flagIn[blockId].Flags, j) == 0) {
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

            inOffset += sizeof(PairType);
        }
    }

    if (finishCallback != nullptr) {
        finishCallback(blockId);
    }
}
