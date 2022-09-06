#!/bin/bash
fileid="10z6vc8OvQ0sLPUH6wzpTmFEImQEBFgzt"
filename="analog-meter_dataset.tar.gz"
html=`curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}"`
curl -Lb ./cookie "https://drive.google.com/uc?export=download&`echo ${html}|grep -Po '(confirm=[a-zA-Z0-9\-_]+)'`&id=${fileid}" -o ${filename}
rm ./cookie
if [ -f $filename ]; then
    tar xvzf $filename
    if [ ! -e dataset ]; then
        mkdir dataset
    fi
    mv analog-meter dataset
    rm $filename
fi
