#include <string>
#include <unordered_map>

#include <memory.h>
#include <stdint.h>

#include "../Settings.h"
#include "../LZSSInterface.h"

#include "../Helper/Helper.h"

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
        // Fill the remaining with a character that occurs often
        memset(blockBuf + blockSize, ' ', DataBlockSize - blockSize);

        // The first character
        auto nflags = 1;
        auto written = 1;

        PUT_BIT(flagOut[i].Flags, 0, 0);
        outBuf[blockOffset] = blockBuf[0];

        // Later on
        for (int j = 1; j < blockSize; ) {
            auto lookbackLength = std::min(WindowSize, j);
            auto lookaheadLength = std::min(MaxEncodeLength, blockSize - j);
            int matchOffset, matchLength;

            if (FindMatch(blockBuf + j - lookbackLength, lookbackLength,
                    blockBuf + j, lookaheadLength,
                    matchOffset, matchLength)) {

                PairType matchPair = (matchOffset << PairLengthBits) | matchLength;
                memcpy(outBuf + blockOffset + written, &matchPair, sizeof(PairType));
                written += sizeof(PairType);

                PUT_BIT(flagOut[i].Flags, nflags, 1);
                j += matchLength;
            } else {
                outBuf[blockOffset + written] = blockBuf[j];
                written += 1;

                PUT_BIT(flagOut[i].Flags, nflags, 0);
                j += 1;
            }

            ++nflags;
        }

        flagOut[i].NumOfFlags = (uint16_t) nflags;
        flagOut[i].CompressedSize = written;
        
        flagSize += SIZE_OF_FLAGS(nflags) + sizeof(CompressFlagBlock::NumOfFlags); // taken by current flag block
        outSize += written;
    }

    return true;
}

bool CPUSingleThreadLZSS::decompress(const uint8_t* inBuf, int inSize, uint8_t* outBuf, int& outSize)
{
}
