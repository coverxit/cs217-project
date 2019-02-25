#include <iostream>
#include <string>
#include <unordered_map>

#include <assert.h>
#include <memory.h>
#include <stdint.h>

#include "../Settings.h"

#include "../Helper/Helper.h"
#include "../LZSSInterface.h"

#include "CPUSingleThreadLZSS.h"

bool CPUSingleThreadLZSS::compress(const uint8_t* inBuf, int inSize,
    uint8_t* outBuf, int& outSize,
    CompressFlagBlock* flagOut, int& flagSize)
{
    auto nBlocks = (inSize - 1) / DataBlockSize + 1;
    outSize = flagSize = 0;

    for (int i = 0; i < nBlocks; ++i) {
        uint8_t blockBuf[DataBlockSize];
        auto blockOffset = i * DataBlockSize;
        auto blockSize = std::min(DataBlockSize, inSize - blockOffset);

        memcpy(blockBuf, inBuf + blockOffset, blockSize);

        // The first character
        auto nFlags = 1;
        auto written = 1;

        PUT_BIT(flagOut[i].Flags, 0, 0);
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

                assert(matchOffset > 0 && matchOffset <= lookbackLength && matchOffset <= WindowSize);
                assert(matchLength >= ReplaceThreshold && matchLength <= lookaheadLength);

                // Due to the bit limit, minus 1 for exact offset and length
                PairType matchPair = ((matchOffset - 1) << PairLengthBits) | (matchLength - 1);
                memcpy(outBuf + blockOffset + written, &matchPair, sizeof(PairType));
                written += sizeof(PairType);

                PUT_BIT(flagOut[i].Flags, nFlags, 1);
                j += matchLength;
            } else {
                outBuf[blockOffset + written] = blockBuf[j];
                written += 1;

                PUT_BIT(flagOut[i].Flags, nFlags, 0);
                j += 1;
            }

            ++nFlags;
            assert(nFlags <= DataBlockSize);
        }

        flagOut[i].NumOfFlags = (uint16_t)nFlags;
        flagOut[i].CompressedSize = written;

        // taken by current flag block
        flagSize += SIZE_OF_FLAGS(nFlags) + sizeof(CompressFlagBlock::NumOfFlags)
            + sizeof(CompressFlagBlock::CompressedSize);
        outSize += written;

        std::cout << "Block " << (i + 1) << "/" << nBlocks << " done." << std::endl;
    }

    return true;
}

bool CPUSingleThreadLZSS::decompress(CompressFlagBlock* flagIn, int nFlagBlocks,
    const uint8_t* inBuf, int inSize, uint8_t* outBuf)
{
    for (int i = 0; i < nFlagBlocks; ++i) {
        uint8_t blockBuf[DataBlockSize];

        auto inOffset = flagIn[i].CompressedOffset;
        auto outOffset = i * DataBlockSize;

        for (int j = 0; j < flagIn[i].NumOfFlags; ++j) {
            if (GET_BIT(flagIn[i].Flags, j) == 0) {
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
                for (int i = 0; i < matchLength; ++i) {
                    outBuf[outOffset] = outBuf[outOffset - matchOffset];
                    ++outOffset;
                }

                inOffset += 2;
            }
        }
    }

    return true;
}
