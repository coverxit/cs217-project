#!/bin/bash

exec=../bin/lzss_nvcc
corpus=(canterbury_corpus large_corpus miscellaneous_corpus)
logs=(cpu_st.comp.log cpu_mt.comp.log gpu.comp.log)
flags=(s m g)
modes=(CPU-ST CPU-MT GPU)

if [ ! -f exec ]; then
    echo "Please compile the program first!"
    exit 1
fi

for l in ${logs[@]}; do
    rm -f ${l}
done

for d in ${corpus[@]}; do
    for f in ${d}/*; do
        if [ -f "${f}" ]; then
            checksum=$(sha1sum ${f} | cut -f1 -d' ')
            echo "Testing ${f}..."
            echo "Original SHA1: ${checksum}"

            for i in "${!flags[@]}"; do
                echo "  Running kernel ${modes[$i]}..."
                rm -f tmp.comp
                rm -f tmp.decomp

                ${exec} c${flags[$i]} ${f} tmp.comp >> ${logs[$i]}
                echo '----------------------------------------------------------------------------------' >> ${logs[$i]}
                comp_checksum=$(sha1sum tmp.comp | cut -f1 -d' ')
                echo "    Compression SHA1:   ${comp_checksum}"

                ${exec} d${flags[$i]} tmp.comp tmp.decomp > /dev/null
                decomp_checksum=$(sha1sum tmp.decomp | cut -f1 -d' ')
                
                if [ "${checksum}" == "${decomp_checksum}" ]; then
                    echo "    Decompression SHA1: ${decomp_checksum}, Match: Yes"
                else
                    echo "    Decompression SHA1: ${decomp_checksum}, Match: No"
                fi
            done
            echo "----------------------------------------------------------------------------"
        fi
    done
done

rm -f tmp.comp
rm -f tmp.decomp
