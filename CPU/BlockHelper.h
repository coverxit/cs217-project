#pragma once

void blockCompress(int blockId, const uint8_t* inBuf, int inSize,
    uint8_t* outBuf, int& outSize,
    CompressFlagBlock* flagOut, int& flagSize,
    std::function<void(int)> finishCallback = nullptr);

void blockDecompress(int blockId, CompressFlagBlock* flagIn, int nFlagBlocks,
    const uint8_t* inBuf, int inSize, uint8_t* outBuf,
    std::function<void(int)> finishCallback = nullptr);
