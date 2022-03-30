export PATH=/opt/tiger/cuda/bin:$PATH
export LD_LIBRARY_PATH=/opt/tiger/cuda/lib64/:$LD_LIBRARY_PATH
export CPATH=/opt/tiger/cuda/include:$CPATH
export CUDA_HOME=/opt/tiger/cuda
export LIBHDFS_OPTS="-Dhadoop.root.logger=${HADOOP_ROOT_LOGGER:-WARN,console}"

hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huangmingkun/tools/anaconda_py37_torch15.tar.gz
mkdir -p ~/anaconda3
tar xf anaconda_py37_torch15.tar.gz -C ~/anaconda3

export PATH=/home/tiger/anaconda3/bin:$PATH

hdfs dfs -get hdfs://haruna/home/byte_arnold_hl_speech_asr/user/huangmingkun/install_pkg ~/pkgs
pip install --no-index --find-links=~/pkgs dataserver
pip install --no-index --find-links=~/pkgs bytespeech
