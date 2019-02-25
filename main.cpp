#include <algorithm>
#include <atomic>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <stdint.h>
#include <string.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <fcntl.h>
#include <unistd.h>

#include "Settings.h"

#include "Helper/Helper.h"
#include "LZSSInterface.h"

#include "ConcurrentStream/ConcurrentStream.hpp"
#include "ConcurrentStream/ConcurrentInputStream.hpp"
#include "ConcurrentStream/ConcurrentOutputStream.hpp"

void compress(AbstractLZSS* lzss, const uint8_t* inBuf, int inSize, const char* outFile)
{
    auto outBuf = new uint8_t[inSize];

    auto nFlagBlocks = (inSize - 1) / DataBlockSize + 1;
    auto flagBlocks = new CompressFlagBlock[nFlagBlocks];
    
    CompressedFileHeader header { DefaultMagic, inSize };
    int outSize, flagSize;

    if (lzss->compress(inBuf, inSize, outBuf, outSize, flagBlocks, flagSize)) {
        std::vector<std::pair<int, int>> offsets;
        std::vector<int> sizes;
        ConcurrentOutputStream outStream(outFile, outSize + flagSize + sizeof(CompressedFileHeader));

        if (outStream) {
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

    auto headerSize = flagSize + sizeof(CompressedFileHeader);
    auto totalOutSize = outSize + headerSize;

    std::cout << "In: " << inSize << " bytes" << std::endl;
    std::cout << "Out: " << totalOutSize << " bytes (Header: ";
    std::cout << headerSize << " bytes, Content: " << outSize << " bytes)" << std::endl;
    std::cout << "Ratio: " << ((float)totalOutSize / inSize) << std::endl;

    delete[] outBuf;
    delete[] flagBlocks;
}

void decompress(AbstractLZSS* lzss, const uint8_t* inBuf, int inSize, const char* outFile)
{
    CompressedFileHeader header = *(CompressedFileHeader*)inBuf;

    if (header.Magic != DefaultMagic) {
        std::cerr << "Magic mismtach (0x" << std::hex << header.Magic;
        std::cerr << " != 0x" << std::hex << DefaultMagic << ")!" << std::endl;
        return;
    }

    auto outSize = header.OriginalSize;
    auto nFlagBlocks = (outSize - 1) / DataBlockSize + 1;
    auto flagBlocks = new CompressFlagBlock[nFlagBlocks];
    auto offset = sizeof(CompressedFileHeader);

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
    if (lzss->decompress(flagBlocks, nFlagBlocks, inBuf + offset, inSize - offset, outBuf)) {
        ConcurrentOutputStream outStream(outFile, outSize);

        if (outStream) {
            outStream.writeNext(outBuf, outSize, std::thread::hardware_concurrency());
            outStream.close();
        }
    }

    std::cout << "In: " << inSize << " bytes (Header: ";
    std::cout << offset << " bytes, Content: " << (inSize - offset) << " bytes)" << std::endl;
    std::cout << "Out: " << outSize << " bytes" << std::endl;
    std::cout << "Ratio: " << (float) outSize / inSize << std::endl;

    delete[] flagBlocks;
    delete[] outBuf;
}

int main(int argc, char const* argv[])
{
    char operation = 0, kernel = 0;

    if (argc != 4) {
        std::cout << "Usage: " << argv[0] << "<options> <input file> <output file>" << std::endl;
        std::cout << "Available options:" << std::endl;
        std::cout << "  Operation:" << std::endl;
        std::cout << "    c - compress" << std::endl;
        std::cout << "    d - decompress" << std::endl;
        std::cout << "  Kernel:" << std::endl;
        std::cout << "    s - CPU single-thread" << std::endl;
        std::cout << "    m - CPU multi-thread" << std::endl;
        std::cout << "    g - GPU CUDA" << std::endl;
        return 1;
    }

    if (strlen(argv[1]) != 2) {
        std::cout << "You must specify two options (one for operation, another for kernel)!" << std::endl;
        return 1;
    }

    for (int i = 0; i < 2; ++i) {
        char opt = argv[1][i];
        if (opt == 'c' || opt == 'd') {
            if (operation == 0) {
                operation = opt;
            } else {
                std::cout << "Only one operation is allowed at the same time!" << std::endl;
                return 1;
            }
        } else if (opt == 's' || opt == 'm' || opt == 'g') {
            if (kernel == 0) {
                kernel = opt;
            } else {
                std::cout << "Only one kernel is allowed at the same time!" << std::endl;
                return 1;
            }
        }
    }

    if (operation == 0 || kernel == 0) {
        std::cout << "You must specify two options (one for operation, another for kernel)!" << std::endl;
        return 1;
    }

    ConcurrentInputStream inStream(argv[2]);
    if (inStream) {
        auto inBuf = new uint8_t[inStream.size()];

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
                lzss = AbstractLZSS::create("CUDA");
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
