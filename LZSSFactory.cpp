#include <string>
#include <unordered_map>

#include <stdint.h>
#include <string.h>

#include "Settings.h"
#include "LZSSInterface.h"

#include "CPU/ST/CPUSingleThreadLZSS.h"
#include "CPU/MT/CPUMultiThreadLZSS.h"
#include "CUDA/CUDALZSS.h"

AbstractLZSS* AbstractLZSS::create(const char* type)
{
    if (!strcmp(type, "CPUST")) {
        return new CPUSingleThreadLZSS;
    } else if (!strcmp(type, "CPUMT")) {
        return new CPUMultiThreadLZSS;
    } else if (!strcmp(type, "CUDA")) {
        return nullptr;
        //return new CUDALZSS;
    }
}
