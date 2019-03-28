#!/bin/bash

# Build custom kernels.
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
g++ -std=c++11 -shared srl_kernels.cc -o srl_kernels.so -I$TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -O2
