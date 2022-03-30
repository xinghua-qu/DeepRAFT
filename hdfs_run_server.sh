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

export TZ=Asia/Shanghai
export PATH=/root/anaconda3/bin:$PATH
export PATH=/home/tiger/anaconda3/bin:$PATH

export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu/:/usr/lib/x86_64-linux-gnu/:/opt/tiger/kaldi_lib/cuda-9.0/lib:/opt/tiger/kaldi_lib/intel/compilers_and_libraries_2017.2.174/linux/compiler/lib/intel64:/opt/tiger/kaldi_lib/intel/compilers_and_libraries_2017.2.174/linux/compiler/lib/intel64_lin:/opt/tiger/kaldi_lib/intel/compilers_and_libraries_2017.2.174/linux/tbb/lib/intel64_lin/gcc4.7:/opt/tiger/kaldi_lib/intel/compilers_and_libraries_2017.2.174/linux/compiler/lib/intel64_lin:/opt/tiger/kaldi_lib/intel/compilers_and_libraries_2017.2.174/linux/mkl/lib/intel64_lin:/opt/tiger/kaldi_lib/speech_kaldi/lib:/opt/tiger/kaldi_lib/speech_openfst:/opt/tiger/kaldi_lib/speech_mkl:$LD_LIBRARY_PATH

if [[ $world_size -gt 1 ]]; then
  echo "mpi mode"
  exec MPIRUN -n $world_size python3 -m dataserver.train --distributed-init-method tcp $configure > data.tr.log 2>&1 &
  exec MPIRUN -n $world_size python3 -m dataserver.valid --distributed-init-method tcp $configure > data.cv.log 2>&1 &
  exec MPIRUN -n $world_size python3 -m bytespeech.train --distributed-init-method tcp $configure
  exit 0
fi

exec python3 -m dataserver.train $configure > data.tr.log 2>&1 &
exec python3 -m dataserver.valid $configure > data.cv.log 2>&1 &
exec python3 -m bytespeech.train $configure
