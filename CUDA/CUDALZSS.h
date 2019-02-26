#pragma once

class CUDALZSS final : public AbstractLZSS {
public:
    std::pair<bool, double> compress(const uint8_t* inBuf, int inSize,
        uint8_t* outBuf, int& outSize,
        CompressFlagBlock* flagOut, int nFlagBlocks, int& flagSize) override;

    std::pair<bool, double> decompress(CompressFlagBlock* flagIn, int nFlagBlocks,
        const uint8_t* inBuf, uint8_t* outBuf) override;
};
