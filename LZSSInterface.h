#pragma once

struct CompressedFileHeader {
    int Magic;
    int OriginalSize;
};

struct CompressFlagBlock {
    uint8_t Flags[DataBlockSize / 8];
    uint16_t NumOfFlags;
    uint16_t CompressedSize;
    uint32_t CompressedOffset; // Not written to file, used during decompression.
                               // Type should be changed, if file size limit raise.
};

class AbstractLZSS {
public:
    virtual ~AbstractLZSS() {}

    virtual std::pair<bool, double> compress(const uint8_t* inBuf, int64_t inSize,
        uint8_t* outBuf, int64_t& outSize,
        CompressFlagBlock* flagOut, int64_t nFlagBlocks, int64_t& flagSize)
        = 0;

    virtual std::pair<bool, double> decompress(CompressFlagBlock* flagIn, int64_t nFlagBlocks,
        const uint8_t* inBuf, int64_t inSize, uint8_t* outBuf, int64_t outSize)
        = 0;

public:
    // Factory method
    static AbstractLZSS* create(const char* type);
};
