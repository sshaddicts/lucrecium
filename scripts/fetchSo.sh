#!/bin/bash

mode=${1}

echo wget https://raw.githubusercontent.com/sshaddicts/opencv/repository/libopencv_java320.so

if [[ $mode == "travis" ]];
then
    echo "sudo mkdir -p /usr/lib"
    echo "sudo mv libopencv_java320.so /usr/lib/opencv_java320.so"
else
    echo "sudo mkdir -p /usr/lib/jni"
    echo "sudo mv libopencv_java320.so /usr/lib/jni/opencv_java320.so"
fi


