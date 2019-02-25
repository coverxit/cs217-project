#include <functional>
#include <iostream>
#include <string>
#include <unordered_map>

#include <memory.h>
#include <stdint.h>

#include "../../Settings.h"

#include "../../LZSSInterface.h"

#include "../BlockHelper.h"
#include "CPUSingleThreadLZSS.h"

bool CPUSingleThreadLZSS::compress(const uint8_t* inBuf, int inSize,
    uint8_t* outBuf, int& outSize,
    CompressFlagBlock* flagOut, int& flagSize)
{
    auto nBlocks = (inSize - 1) / DataBlockSize + 1;
    outSize = flagSize = 0;

    for (int i = 0; i < nBlocks; ++i) {
        blockCompress(i, inBuf, inSize, outBuf, outSize, flagOut, flagSize,
            [nBlocks](int blockId) {
                if ((blockId + 1) % 100 == 0) {
                    std::cout << "Block " << (blockId + 1) << "/" << nBlocks << " done." << std::endl;
                }
            });
    }

    return true;
}

bool CPUSingleThreadLZSS::decompress(CompressFlagBlock* flagIn, int nFlagBlocks,
    const uint8_t* inBuf, int inSize, uint8_t* outBuf)
{
    for (int i = 0; i < nFlagBlocks; ++i) {
        blockDecompress(i, flagIn, nFlagBlocks, inBuf, inSize, outBuf);
    }

    return true;
}
