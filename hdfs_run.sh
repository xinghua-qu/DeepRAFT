#!/bin/bash

# -d 1: means use debug mode, the arnold machine will sleep for user to debug
#    0: means normal mode
# -n 1: means single worker; >1 means multi-worker or multi nodes
# -f the python args


THIS_DIR="$( cd "$( dirname "$0" )" && pwd )"
cd $THIS_DIR

while getopts "d:n:f:" arg
do
        case $arg in
             d)
                echo "debug mode:$OPTARG" #参数存在$OPTARG中
                debug=$OPTARG
                ;;
             n)
                echo "mpi world_size:$OPTARG"
                world_size=$OPTARG
                ;;
             f)
                echo "trian.py configure:$OPTARG"
                configure=$OPTARG
                ;;
             ?)
            echo "unkonw argument"
        exit 1
        ;;
        esac
done

if [[ $debug = 1 ]]; then
  echo 'begin to debug on arnold'
  sleep 100000000
  exit 0
fi

bash hdfs_docker_build.sh $configure
echo "waiting for all workers finish docker build"

hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_speech_asr/user/xinghua/datasets/mirflickr25k.zip
unzip mirflickr25k.zip
rm -f mirflickr25k.zip

python3 train.py $configure

hdfs dfs -put -f results/* hdfs://haruna/home/byte_arnold_hl_speech_asr/user/xinghua/speech-security/AF-Stamp/