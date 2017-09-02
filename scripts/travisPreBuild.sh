#!/bin/bash

sudo apt-get update -y
sudo apt-get install build-essential -y
sudo apt-get install cmake git libgtk2.0-dev -y pkg-config libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev -y

#wget http://security.ubuntu.com/ubuntu/pool/universe/liba/libav/libav-tools_9.20-0ubuntu0.14.04.1_amd64.deb
wget http://mt.archive.ubuntu.com/ubuntu/pool/main/g/gstreamer0.10/libgstreamer0.10-0_0.10.36-1.2ubuntu3_amd64.deb

#sudo apt install ./libav-tools_9.20-0ubuntu0.14.04.1_amd64.deb
sudo apt-get install ./libgstreamer0.10-0_0.10.36-1.2ubuntu3_amd64.deb

#rm libav-tools_9.20-0ubuntu0.14.04.1_amd64.deb
rm libgstreamer0.10-0_0.10.36-1.2ubuntu3_amd64.deb
    