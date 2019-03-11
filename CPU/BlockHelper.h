#pragma once

void BlockCompress(int blockId, const uint8_t* inBuf, int inSize,
    uint8_t* outBuf, int& outSize,
    CompressFlagBlock* flagOut, int& flagSize,
    std::function<void(int)> finishCallback = nullptr);

void BlockDecompress(int blockId, CompressFlagBlock* flagIn,
    const uint8_t* inBuf, uint8_t* outBuf,
    std::function<void(int)> finishCallback = nullptr);
