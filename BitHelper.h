#pragma once

#define GET_BIT(flags, bit) \
    (((flags)[(bit) / 8] >> ((bit) % 8)) & 0x1)

#define PUT_BIT(flags, bit, value) \
    PUT_BIT_##value(flags, bit)

#define PUT_BIT_0(flags, bit) \
    ((flags)[(bit) / 8] &= ~(1 << ((bit) % 8)))

#define PUT_BIT_1(flags, bit) \
    ((flags)[(bit) / 8] |= (1 << ((bit) % 8)))

#define SIZE_OF_FLAGS(numFlags) \
    ((numFlags) - 1) / 8 + 1

#define CHUNK_LOW(idx, total, nThreads) \
    ((idx) * (total) / (nThreads))

#define CHUNK_SIZE(idx, total, nThreads) \
    (CHUNK_LOW((idx) + 1, total, nThreads) - CHUNK_LOW(idx, total, nThreads))
