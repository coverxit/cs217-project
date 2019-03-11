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

std::pair<bool, double> CPUSingleThreadLZSS::compress(const uint8_t* inBuf, int64_t inSize,
    uint8_t* outBuf, int64_t& outSize,
    CompressFlagBlock* flagOut, int64_t nFlagBlocks, int64_t& flagSize)
{
    Timer timer;

    outSize = flagSize = 0;
    for (int64_t i = 0; i < nFlagBlocks; ++i) {
        BlockCompress(i, inBuf, inSize, outBuf, outSize, flagOut, flagSize,
            [nFlagBlocks](int64_t blockId) {
                if ((blockId + 1) % 100 == 0) {
                    printf("Block %" PRId64 "/%" PRId64 " done.\n", blockId + 1, nFlagBlocks);
                }
            });
    }

    return std::make_pair(true, timer.end());
}

std::pair<bool, double> CPUSingleThreadLZSS::decompress(CompressFlagBlock* flagIn, int64_t nFlagBlocks,
    const uint8_t* inBuf, int64_t inSize, uint8_t* outBuf, int64_t outSize)
{
    Timer timer;

    for (int64_t i = 0; i < nFlagBlocks; ++i) {
        BlockDecompress(i, flagIn, inBuf, outBuf);
    }

    return std::make_pair(true, timer.end());
}
