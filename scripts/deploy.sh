#!/bin/bash

sudo apt-get install build-essential -y
sudo apt-get install cmake git libgtk2.0-dev -y pkg-config libavcodec-dev libavformat-dev libswscale-dev 
sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev -y

mkdir opencv-sources
cd opencv-sources
git clone https://github.com/opencv/opencv.git

cd opencv
mkdir release
cd release

cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/lib/ -D BUILD_SHARED_LIBS=OFF ..

make