#pragma once

void BlockCompress(int64_t blockId, const uint8_t* inBuf, int64_t inSize,
    uint8_t* outBuf, int64_t& outSize,
    CompressFlagBlock* flagOut, int64_t& flagSize,
    std::function<void(int64_t)> finishCallback = nullptr);

void BlockDecompress(int64_t blockId, CompressFlagBlock* flagIn,
    const uint8_t* inBuf, uint8_t* outBuf,
    std::function<void(int64_t)> finishCallback = nullptr);
