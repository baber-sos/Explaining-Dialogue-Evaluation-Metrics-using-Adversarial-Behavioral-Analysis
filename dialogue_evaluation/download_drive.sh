#!/bin/bash
fileid="### file id ###"
filename="gpt-pretrained2"
curl -c ./cookie -s -L "https://drive.google.com/u/0/uc?id=1kNBYgIucYRBXYdCnn8CQ5mSRLYKuL4Y2&export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
