#include <algorithm>

#include <assert.h>
#include <stdint.h>

#include "../Settings.h"
#include "Helper.h"

#ifdef __NVCC__
__device__
#endif
bool FindMatch(const uint8_t* searchBuf, int searchBufSize,
    const uint8_t* matchBuffer, int matchBufferSize,
    int& matchOffset, int& matchLength)
{
    matchOffset = -1;
    matchLength = 0;

    for (int i = 0; i < searchBufSize; ++i) {
        int currentLength = 0;

        for (int j = 0, start = i; j < matchBufferSize; ++j, ++start) {
            if (searchBuf[start] == matchBuffer[j]) {
                ++currentLength;
            } else {
                break;
            }
        }

        if (currentLength > matchLength) {
            matchLength = currentLength;
            matchOffset = i;
        }
    }

    return matchLength >= ReplaceThreshold;
}
