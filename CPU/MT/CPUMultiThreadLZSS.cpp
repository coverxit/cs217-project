#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <stdio.h>
#include <memory.h>
#include <stdint.h>

#include "../../Settings.h"

#include "../../LZSSInterface.h"

#include "../BlockHelper.h"
#include "CPUMultiThreadLZSS.h"

bool CPUMultiThreadLZSS::compress(const uint8_t* inBuf, int inSize,
    uint8_t* outBuf, int& outSize,
    CompressFlagBlock* flagOut, int& flagSize)
{
    auto nBlocks = (inSize - 1) / DataBlockSize + 1;
    std::atomic_int atomicOutSize(0), atomicFlagSize(0), atomicBlocksDone(0);
    std::mutex outputMutex;

    auto nThreads = std::thread::hardware_concurrency();

    // Too many threads?
    if (nThreads > nBlocks) {
        nThreads = nBlocks;
    }

    // Left over?
    if (nBlocks % nThreads) {
        ++nThreads;
    }

    std::vector<std::thread> threads;
    threads.reserve(nThreads);

    // Process block in parallel
    for (int i = 0; i < nThreads; ++i) {
        auto chunk = (nBlocks - 1) / nThreads + 1;
        auto offset = chunk * i;
        auto length = std::min(chunk, nBlocks - offset);

        threads.emplace_back([&, offset, length]() {
            int tempOutSize = 0, tempFlagSize = 0;

            for (int j = offset; j < offset + length; ++j) {
                blockCompress(j, inBuf, inSize, outBuf, tempOutSize, flagOut, tempFlagSize,
                    [nBlocks, &outputMutex, &atomicBlocksDone](int blockId) {
                        ++atomicBlocksDone;

                        if (atomicBlocksDone % 100 == 0) {
                            outputMutex.lock();
                            printf("Block %d/%d done.\n", atomicBlocksDone.load(), nBlocks);
                            outputMutex.unlock();
                        }
                    });
            }

            atomicOutSize += tempOutSize;
            atomicFlagSize += tempFlagSize;
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    outSize = atomicOutSize;
    flagSize = atomicFlagSize;
    return true;
}

bool CPUMultiThreadLZSS::decompress(CompressFlagBlock* flagIn, int nFlagBlocks,
    const uint8_t* inBuf, int inSize, uint8_t* outBuf)
{
    auto nThreads = std::thread::hardware_concurrency();

    // Too many threads?
    if (nThreads > nFlagBlocks) {
        nThreads = nFlagBlocks;
    }

    // Left over?
    if (nFlagBlocks % nThreads) {
        ++nThreads;
    }

    std::vector<std::thread> threads;
    threads.reserve(nThreads);

    // Process block in parallel
    for (int i = 0; i < nThreads; ++i) {
        auto chunk = (nFlagBlocks - 1) / nThreads + 1;
        auto offset = chunk * i;
        auto length = std::min(chunk, nFlagBlocks - offset);

        threads.emplace_back([&, offset, length]() {
            for (int j = offset; j < offset + length; ++j) {
                blockDecompress(j, flagIn, nFlagBlocks, inBuf, inSize, outBuf);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    return true;
}
