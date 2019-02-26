#include <functional>

#include <memory.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#include "../../Settings.h"

#include "../../LZSSInterface.h"
#include "../../TimerHelper.hpp"

#include "../BlockHelper.h"
#include "CPUSingleThreadLZSS.h"

std::pair<bool, double> CPUSingleThreadLZSS::compress(const uint8_t* inBuf, int inSize,
    uint8_t* outBuf, int& outSize,
    CompressFlagBlock* flagOut, int nFlagBlocks, int& flagSize)
{
    Timer timer;

    outSize = flagSize = 0;
    for (int i = 0; i < nFlagBlocks; ++i) {
        blockCompress(i, inBuf, inSize, outBuf, outSize, flagOut, flagSize,
            [nFlagBlocks](int blockId) {
                if ((blockId + 1) % 100 == 0) {
                    printf("Block %d/%d done.\n", blockId + 1, nFlagBlocks);
                }
            });
    }

    return std::make_pair(true, timer.end());
}

std::pair<bool, double> CPUSingleThreadLZSS::decompress(CompressFlagBlock* flagIn, int nFlagBlocks,
    const uint8_t* inBuf, int inSize, uint8_t* outBuf)
{
    Timer timer;

    for (int i = 0; i < nFlagBlocks; ++i) {
        blockDecompress(i, flagIn, nFlagBlocks, inBuf, inSize, outBuf);
    }

    return std::make_pair(true, timer.end());
}
