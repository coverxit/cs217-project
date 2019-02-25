#include <string>
#include <unordered_map>

#include <stdint.h>
#include <string.h>

#include "Settings.h"
#include "LZSSInterface.h"

#include "CPUMT/CPUMultiThreadLZSS.h"
#include "CPUST/CPUSingleThreadLZSS.h"
#include "CUDA/CUDALZSS.h"

AbstractLZSS* AbstractLZSS::create(const char* type)
{
    if (!strcmp(type, "CPUST")) {
        return new CPUSingleThreadLZSS;
    } else if (!strcmp(type, "CPUMT")) {
        return nullptr;
        //return new CPUMultiThreadLZSS;
    } else if (!strcmp(type, "CUDA")) {
        return nullptr;
        //return new CUDALZSS;
    }
}
