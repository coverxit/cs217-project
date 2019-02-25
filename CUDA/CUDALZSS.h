#pragma once

class CUDALZSS final : public AbstractLZSS {
public:
    bool compress(const uint8_t* inBuf, int inSize,
        uint8_t* outBuf, int& outSize,
        CompressFlagBlock* flagOut, int& flagSize) override;
        
    bool decompress(const uint8_t* inBuf, int inSize, uint8_t* outBuf, int& outSize) override;
};
