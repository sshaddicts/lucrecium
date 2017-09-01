#!/bin/bash

wget http://security.ubuntu.com/ubuntu/pool/universe/liba/libav/libavcodec-extra-54_9.20-0ubuntu0.14.04.1_amd64.deb
wget http://security.ubuntu.com/ubuntu/pool/universe/liba/libav/libavformat54_9.20-0ubuntu0.14.04.1_amd64.deb


sudo apt install ./libavcodec-extra-54_9.20-0ubuntu0.14.04.1_amd64.deb
sudo apt install ./libavformat54_9.20-0ubuntu0.14.04.1_amd64.deb


rm libavcodec-extra-54_9.20-0ubuntu0.14.04.1_amd64.deb
rm libavformat54_9.20-0ubuntu0.14.04.1_amd64.deb
