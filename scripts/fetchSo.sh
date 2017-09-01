#!/bin/bash
wget https://raw.githubusercontent.com/sshaddicts/opencv/repository/libopencv_java320.so

sudo mkdir -p /usr/lib/jni
sudo mv libopencv_java320.so /usr/lib/jni/opencv_java320.so
