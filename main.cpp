#include <algorithm>
#include <atomic>
#include <thread>
#include <vector>

#include <stdio.h>
#include <stdint.h>
#include <memory.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <fcntl.h>
#include <unistd.h>

#ifndef GCC_TARGET
#include <cuda_runtime_api.h>
#endif

#include "Settings.h"

#include "BitHelper.h"
#include "LZSSInterface.h"

#include "ConcurrentStream/ConcurrentStream.hpp"

#include "ConcurrentStream/ConcurrentInputStream.hpp"
#include "ConcurrentStream/ConcurrentOutputStream.hpp"

void compress(AbstractLZSS* lzss, const uint8_t* inBuf, int inSize, const char* outFile)
{
    auto outBuf = new uint8_t[inSize];

    auto nFlagBlocks = (inSize - 1) / DataBlockSize + 1;
    auto flagBlocks = new CompressFlagBlock[nFlagBlocks];
    
    CompressedFileHeader header{ DefaultMagic, inSize };
    int outSize, flagSize;

    printf("Compressing...\n");
    memset(flagBlocks, 0, sizeof(CompressFlagBlock) * nFlagBlocks);

    auto retVal = lzss->compress(inBuf, inSize, outBuf, outSize, flagBlocks, nFlagBlocks, flagSize);
    if (retVal.first) {
        std::vector<std::pair<int, int>> offsets;
        std::vector<int> sizes;
        ConcurrentOutputStream outStream(outFile, outSize + flagSize + sizeof(CompressedFileHeader));

        if (outStream) {
            printf("Writing to file %s...\n", outFile);
            outStream.writeNext((uint8_t*)&header, sizeof(CompressedFileHeader), 1);

            for (int i = 0, offset = 0; i < nFlagBlocks; ++i) {
                outStream.writeNext((uint8_t*)&flagBlocks[i].NumOfFlags, sizeof(CompressFlagBlock::NumOfFlags), 1);
                outStream.writeNext((uint8_t*)&flagBlocks[i].CompressedSize, sizeof(CompressFlagBlock::CompressedSize), 1);
                outStream.writeNext(flagBlocks[i].Flags, SIZE_OF_FLAGS(flagBlocks[i].NumOfFlags), 1);

                offsets.push_back(std::make_pair(offset, DataBlockSize * i));
                sizes.push_back(flagBlocks[i].CompressedSize);

                offset += flagBlocks[i].CompressedSize;
            }

            outStream.writeNext(outBuf, offsets, sizes, std::thread::hardware_concurrency());
            outStream.close();
        }
    }

    auto headerSize = flagSize + (int)sizeof(CompressedFileHeader);
    auto totalOutSize = outSize + headerSize;

    printf("============== Statistics ==============\n");
    printf("In:            %10d bytes\n", inSize);
    printf("Out:           %10d bytes\n", totalOutSize);
    printf(" - Header:     %10d bytes\n", headerSize);
    printf(" - Content:    %10d bytes\n", outSize);
    printf(" - # Blocks:   %10d\n", nFlagBlocks);
    printf("Ratio:         %10.6f\n", (float) inSize / totalOutSize);
    printf("Time (Kernel): %10.6f secs\n", retVal.second);

    delete[] outBuf;
    delete[] flagBlocks;
}

void decompress(AbstractLZSS* lzss, const uint8_t* inBuf, int inSize, const char* outFile)
{
    CompressedFileHeader header = *(CompressedFileHeader*)inBuf;

    printf("Decompressing...\n");
    if (header.Magic != DefaultMagic) {
        fprintf(stderr, "Magic mismatch (0x%x != 0x%x)!\n", header.Magic, DefaultMagic);
        return;
    }

    auto outSize = header.OriginalSize;
    auto nFlagBlocks = (outSize - 1) / DataBlockSize + 1;
    auto flagBlocks = new CompressFlagBlock[nFlagBlocks];
    int offset = sizeof(CompressedFileHeader);

    memset(flagBlocks, 0, sizeof(CompressFlagBlock) * nFlagBlocks);

    for (int i = 0, dataOffset = 0; i < nFlagBlocks; ++i) {
        flagBlocks[i].NumOfFlags = *(uint16_t*)(inBuf + offset);
        offset += sizeof(CompressFlagBlock::NumOfFlags);

        flagBlocks[i].CompressedSize = *(uint16_t*)(inBuf + offset);
        offset += sizeof(CompressFlagBlock::CompressedSize);

        flagBlocks[i].CompressedOffset = dataOffset;
        memcpy(flagBlocks[i].Flags, inBuf + offset, SIZE_OF_FLAGS(flagBlocks[i].NumOfFlags));

        offset += SIZE_OF_FLAGS(flagBlocks[i].NumOfFlags);
        dataOffset += flagBlocks[i].CompressedSize;
    }

    auto outBuf = new uint8_t[outSize];
    auto retVal = lzss->decompress(flagBlocks, nFlagBlocks, inBuf + offset, inSize - offset, outBuf, outSize);
    if (retVal.first) {
        ConcurrentOutputStream outStream(outFile, outSize);

        if (outStream) {
            printf("Writing to file %s...\n", outFile);
            outStream.writeNext(outBuf, outSize, std::thread::hardware_concurrency());
            outStream.close();
        }
    }

    printf("============== Statistics ==============\n");
    printf("In:            %10d bytes\n", inSize);
    printf(" - Header:     %10d bytes\n", offset);
    printf(" - Content:    %10d bytes\n", inSize - offset);
    printf(" - # Blocks:   %10d\n", nFlagBlocks);
    printf("Out:           %10d bytes\n", outSize);
    printf("Ratio:         %10.6f\n", (float) outSize / inSize);
    printf("Time (Kernel): %10.6f s\n", retVal.second);

    delete[] flagBlocks;
    delete[] outBuf;
}

int main(int argc, char const* argv[])
{
    char operation = 0, kernel = 0;
    int gpus = 0;

    printf("CS 217 Final Project: CUDA version of LZSS algorithm\n");
    printf("Copyright (C) 2019 Renjie Wu, Tong Shen, Zhihui Shao\n");
    printf("Build on " __DATE__ " " __TIME__ ".\n\n");

    printf("System information:\n");
    printf("- # CPU cores in this system: %u\n", std::thread::hardware_concurrency());

#ifndef GCC_TARGET
    cudaGetDeviceCount(&gpus);
    printf("- # GPUs installed in this sytem: %u\n", gpus);
#endif
    
    printf("\n");
    if (argc != 4) {
        printf("Usage: %s <options> <input file> <output file>\n", argv[0]);
        printf("Available options:\n");
        printf("  Operation:\n");
        printf("    c - compress\n");
        printf("    d - decompress\n");
        printf("  Kernel:\n");
        printf("    s - CPU single-thread\n");
        printf("    m - CPU multi-thread\n");
        printf("    g - GPU CUDA\n");
        return 1;
    }

    if (strlen(argv[1]) != 2) {
        fprintf(stderr, "You must specify two options (one for operation, another for kernel)!\n");
        return 1;
    }

    for (int i = 0; i < 2; ++i) {
        char opt = argv[1][i];
        if (opt == 'c' || opt == 'd') {
            if (operation == 0) {
                operation = opt;
            } else {
                fprintf(stderr, "Only one operation is allowed at the same time!\n");
                return 1;
            }
        } else if (opt == 's' || opt == 'm' || opt == 'g') {
            if (kernel == 0) {
                kernel = opt;
            } else {
                fprintf(stderr, "Only one kernel is allowed at the same time!\n");
                return 1;
            }
        }
    }

    if (operation == 0 || kernel == 0) {
        fprintf(stderr, "You must specify two options (one for operation, another for kernel)!\n");
        return 1;
    }

    ConcurrentInputStream inStream(argv[2]);
    if (inStream) {
        auto inBuf = new uint8_t[inStream.size()];

        printf("Reading from file %s\n", argv[2]);
        if (inStream.read(inBuf, inStream.size(), std::thread::hardware_concurrency())) {
            AbstractLZSS* lzss = nullptr;
            switch (kernel) {
            case 's':
                lzss = AbstractLZSS::create("CPUST");
                break;

            case 'm':
                lzss = AbstractLZSS::create("CPUMT");
                break;

            case 'g':
#ifdef GCC_TARGET
                fprintf(stderr, "Please compile with target `nvcc` to launch CUDA kernel!\n");
                
                delete[] inBuf;
                inStream.close();
                return 1;
#else
                lzss = AbstractLZSS::create("CUDA");
#endif
                break;
            }

            // lzss is guaranteed not to be nullptr at this point.
            switch (operation) {
            case 'c':
                compress(lzss, inBuf, inStream.size(), argv[3]);
                break;

            case 'd':
                decompress(lzss, inBuf, inStream.size(), argv[3]);
                break;
            }
        }

        delete[] inBuf;
        inStream.close();
    }

    return 0;
}
