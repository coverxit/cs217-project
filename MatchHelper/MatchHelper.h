#pragma once

#ifdef __NVCC__
__device__
#endif
bool FindMatch(const uint8_t* searchBuf, int searchBufSize,
    const uint8_t* matchBuffer, int matchBufferSize,
    int& matchOffset, int& matchLength);
