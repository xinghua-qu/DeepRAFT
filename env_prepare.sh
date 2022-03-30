#!/bin/bash

export http_proxy=10.20.47.147:3128  https_proxy=10.20.47.147:3128 no_proxy=code.byted.org

hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_speech_asr/user/xinghua/datasets/mirflickr25k.zip
unzip mirflickr25k.zip
rm -f mirflickr25k.zip

python3 train.py 
