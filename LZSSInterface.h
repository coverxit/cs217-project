#pragma once

struct CompressedFileHeader {
    int Magic;
    int OriginalSize;
};

struct CompressFlagBlock {
    uint8_t Flags[DataBlockSize / 8];
    uint16_t NumOfFlags;
    uint16_t CompressedSize;
    uint16_t CompressedOffset; // Not written to file, used during decompression

    CompressFlagBlock()
    {
        memset(Flags, 0, DataBlockSize / 8);
    }
};

class AbstractLZSS {
public:
    virtual ~AbstractLZSS() {}

    virtual bool compress(const uint8_t* inBuf, int inSize,
        uint8_t* outBuf, int& outSize,
        CompressFlagBlock* flagOut, int& flagSize)
        = 0;

    virtual bool decompress(CompressFlagBlock* flagIn, int nFlagBlocks,
        const uint8_t* inBuf, int inSize, uint8_t* outBuf)
        = 0;

public:
    // Factory method
    static AbstractLZSS* create(const char* type);
};

