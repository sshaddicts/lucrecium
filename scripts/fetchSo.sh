#!/bin/bash

mode=${1}

wget https://raw.githubusercontent.com/sshaddicts/opencv/repository/libopencv_java320.so

if [[ $mode == "travis" ]];
then
    sudo mkdir -p /usr/lib
    sudo mv libopencv_java320.so /usr/lib/opencv_java320.so

    ls /urs/lib | grep opencv_
else
    sudo mkdir -p /usr/lib/jni
    sudo mv libopencv_java320.so /usr/lib/jni/opencv_java320.so
fi


