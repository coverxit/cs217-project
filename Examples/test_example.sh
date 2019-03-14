#!/bin/bash

exec=../bin/lzss_nvcc
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
            for i in "${!flags[@]}"; do
                echo "[${modes[$i]}] Testing ${f}..."

                rm -f tmp.comp
                rm -f tmp.decomp

                ${exec} c${flags[$i]} ${f} tmp.comp >> ${logs[$i]}
                echo '----------------------------------------------------------------------------------' >> ${logs[$i]}

                ${exec} d${flags[$i]} tmp.comp tmp.decomp > /dev/null
                diff ${f} tmp.decomp
            done
        fi
    done
done

rm -f tmp.comp
rm -f tmp.decomp
