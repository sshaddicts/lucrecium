#!/bin/bash

mode=${1}

#wget https://raw.githubusercontent.com/sshaddicts/opencv/repository/libopencv_java320.so

if [[ $mode == "travis" ]];
then
    sudo mv 3rdparty/libopencv_java320.so /usr/lib/libopencv_java320.so
else
    sudo mkdir -p /usr/lib/jni
    sudo mv libopencv_java320.so /usr/lib/jni/libopencv_java320.so
fi


