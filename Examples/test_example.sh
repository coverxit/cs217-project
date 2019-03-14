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
            checksum=$(sha1sum ${f} | cut -f1 -d' ')
            echo "Testing ${f}... SHA1: ${checksum}"

            for i in "${!flags[@]}"; do
                rm -f tmp.comp
                rm -f tmp.decomp

                ${exec} c${flags[$i]} ${f} tmp.comp >> ${logs[$i]}
                echo '----------------------------------------------------------------------------------' >> ${logs[$i]}
                comp_checksum=$(sha1sum tmp.comp | cut -f1 -d' ')
                echo "[${modes[$i]}] Compression SHA1: ${comp_checksum}"

                ${exec} d${flags[$i]} tmp.comp tmp.decomp > /dev/null
                decomp_checksum=$(sha1sum tmp.decomp | cut -f1 -d' ')
                
                if [ "${checksum}" - eq "${decomp_checksum}" ]; then
                    echo "[${modes[$i]}] Decompression SHA1: ${decomp_checksum}, Match: Yes"
                else
                    echo "[${modes[$i]}] Decompression SHA1: ${decomp_checksum}, Match: No"
                fi
            done
        fi
    done
done

rm -f tmp.comp
rm -f tmp.decomp
