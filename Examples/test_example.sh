#!/bin/bash

exec=../bin/lzss_nvcc
corpus=(canterbury_corpus large_corpus miscellaneous_corpus)
logs=(cpu_st.comp.log cpu_mt.comp.log gpu.comp.log)
flags=(s m g)
modes=(CPU-ST CPU-MT GPU)
tmp_comp=tmp.comp
tmp_decomp=tmp.decomp

if [ ! -f ${exec} ]; then
    echo "Please compile the program first!"
    exit 1
fi

for l in ${logs[@]}; do
    rm -f ${l}
done

for d in ${corpus[@]}; do
    for f in ${d}/*; do
        if [ -f "${f}" ]; then
            echo "Testing ${f}..."
            
            checksum=$(sha1sum ${f} | cut -f1 -d' ')
            echo "Original SHA1: ${checksum}"

            for i in "${!flags[@]}"; do
                echo "  Running kernel ${modes[$i]}..."
                rm -f ${tmp_comp}
                rm -f ${tmp_decomp}

                ${exec} c${flags[$i]} ${f} ${tmp_comp} >> ${logs[$i]}
                echo '----------------------------------------------------------------------------------' >> ${logs[$i]}
                comp_checksum=$(sha1sum ${tmp_comp} | cut -f1 -d' ')
                echo "    Compression SHA1:   ${comp_checksum}"

                ${exec} d${flags[$i]} ${tmp_comp} ${tmp_decomp} > /dev/null
                decomp_checksum=$(sha1sum ${tmp_decomp} | cut -f1 -d' ')
                
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

rm -f ${tmp_comp}
rm -f ${tmp_decomp}
