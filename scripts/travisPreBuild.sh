#!bin/bash

sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
sudo apt-get update
sudo apt-get install gcc-4.9 -y
sudo apt-get upgrade libstdc++6 -y

