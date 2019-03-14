#!/bin/bash

corpus=(canterbury_corpus large_corpus miscellaneous_corpus)
logs=(cpu_st.comp.log cpu_mt.comp.log gpu.comp.log)
flags=(s m g)
modes=(CPU-ST CPU-MT GPU)

for l in ${logs[@]}; do
    rm -f ${l}
done

for d in ${corpus[@]}; do
    for f in ${d}/*; do
        if [ -f "${f}" ]; then
            for i in "${!logs[@]}"; do
                echo "Running on ${f} through ${modes[$i]}..."

                rm -f tmp.comp
                rm -f tmp.decomp

                ../bin/lzss_nvcc c${flags[$i]} ${f} tmp.comp >> ${logs[$i]}
                echo '------------------------------------------------------------' >> ${logs[$i]}

                ../bin/lzss_nvcc d${flags[$i]} tmp.comp tmp.decomp > /dev/null
                diff ${f} tmp.decomp
            done
        fi
    done
done
