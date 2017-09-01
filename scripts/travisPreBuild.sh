#!/bin/bash

wget http://security.ubuntu.com/ubuntu/pool/main/g/gcc-5/libstdc++6_5.4.0-6ubuntu1~16.04.4_amd64.deb

sudo dpkg -i libstdc++6_5.4.0-6ubuntu1~16.04.4_amd64.deb

rm libstdc++6_5.4.0-6ubuntu1~16.04.4_amd64.deb
