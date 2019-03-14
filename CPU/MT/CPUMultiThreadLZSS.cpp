#include <atomic>
#include <thread>
#include <vector>

#include <math.h>
#include <memory.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#include "../../Settings.h"

#include "../../LZSSInterface.h"
#include "../../TimerHelper.hpp"

#include "../BlockHelper.h"
#include "CPUMultiThreadLZSS.h"

std::pair<bool, double> CPUMultiThreadLZSS::compress(const uint8_t* inBuf, int inSize,
    uint8_t* outBuf, int& outSize,
    CompressFlagBlock* flagOut, int nFlagBlocks, int& flagSize)
{
    Timer timer(false);
    
    std::atomic_int atomicOutSize(0), atomicFlagSize(0), atomicBlocksDone(0);

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
    auto chunk = (size_t) floor((double) nFlagBlocks / nThreads);
    for (int i = 0; i < nThreads; ++i) {
        auto offset = chunk * i;
        auto length = std::min(chunk, nFlagBlocks - offset);

        threads.emplace_back([&, offset, length]() {
            int tempOutSize = 0, tempFlagSize = 0;

            for (int j = offset; j < offset + length; ++j) {
                BlockCompress(j, inBuf, inSize, outBuf, tempOutSize, flagOut, tempFlagSize,
                    [nFlagBlocks, &atomicBlocksDone](int blockId) {
                        // Fetch and add
                        auto fetch = atomicBlocksDone.fetch_add(1) + 1;

                        if (fetch % 100 == 0) {
                            printf("Block %d/%d done.\n", atomicBlocksDone.load(), nFlagBlocks);
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

std::pair<bool, double> CPUMultiThreadLZSS::decompress(CompressFlagBlock* flagIn, int nFlagBlocks,
    const uint8_t* inBuf, int inSize, uint8_t* outBuf, int outSize)
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
    auto chunk = (size_t) floor((double) nFlagBlocks / nThreads);
    for (int i = 0; i < nThreads; ++i) {
        auto offset = chunk * i;
        auto length = std::min(chunk, nFlagBlocks - offset);

        threads.emplace_back([&, offset, length]() {
            for (int j = offset; j < offset + length; ++j) {
                BlockDecompress(j, flagIn, inBuf, outBuf);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    return std::make_pair(true, timer.end());
}
