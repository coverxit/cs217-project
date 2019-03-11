#pragma once

class CUDALZSS final : public AbstractLZSS {
public:
    std::pair<bool, double> compress(const uint8_t* inBuf, int64_t inSize,
        uint8_t* outBuf, int64_t& outSize,
        CompressFlagBlock* flagOut, int64_t nFlagBlocks, int64_t& flagSize) override;

    std::pair<bool, double> decompress(CompressFlagBlock* flagIn, int64_t nFlagBlocks,
        const uint8_t* inBuf, int64_t inSize, uint8_t* outBuf, int64_t outSize) override;
};
