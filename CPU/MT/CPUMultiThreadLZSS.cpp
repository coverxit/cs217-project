#include <atomic>
#include <thread>
#include <vector>

#include <memory.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#include "../../Settings.h"

#include "../../LZSSInterface.h"
#include "../../TimerHelper.hpp"

#include "../BlockHelper.h"
#include "CPUMultiThreadLZSS.h"

std::pair<bool, double> CPUMultiThreadLZSS::compress(const uint8_t* inBuf, int64_t inSize,
    uint8_t* outBuf, int64_t& outSize,
    CompressFlagBlock* flagOut, int64_t nFlagBlocks, int64_t& flagSize)
{
    Timer timer(false);
    
    std::atomic_int64_t atomicOutSize(0), atomicFlagSize(0), atomicBlocksDone(0);

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
    timer.begin();
    for (int i = 0; i < nThreads; ++i) {
        int64_t chunk = (nFlagBlocks - 1) / nThreads + 1;
        int64_t offset = chunk * i;
        int64_t length = std::min(chunk, nFlagBlocks - offset);

        threads.emplace_back([&, offset, length]() {
            int64_t tempOutSize = 0, tempFlagSize = 0;

            for (int64_t j = offset; j < offset + length; ++j) {
                BlockCompress(j, inBuf, inSize, outBuf, tempOutSize, flagOut, tempFlagSize,
                    [nFlagBlocks, &atomicBlocksDone](int64_t blockId) {
                        // Fetch and add
                        int64_t fetch = atomicBlocksDone.fetch_add(1) + 1;

                        if (fetch % 100 == 0) {
                            printf("Block %" PRId64 "/%" PRId64 " done.\n", atomicBlocksDone.load(), nFlagBlocks);
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

    return std::make_pair(true, timer.end());
}

std::pair<bool, double> CPUMultiThreadLZSS::decompress(CompressFlagBlock* flagIn, int64_t nFlagBlocks,
    const uint8_t* inBuf, int64_t inSize, uint8_t* outBuf, int64_t outSize)
{
    Timer timer(false);
    
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
    timer.begin();
    for (int i = 0; i < nThreads; ++i) {
        int64_t chunk = (nFlagBlocks - 1) / nThreads + 1;
        int64_t offset = chunk * i;
        int64_t length = std::min(chunk, nFlagBlocks - offset);

        threads.emplace_back([&, offset, length]() {
            for (int64_t j = offset; j < offset + length; ++j) {
                BlockDecompress(j, flagIn, inBuf, outBuf);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    return std::make_pair(true, timer.end());
}
