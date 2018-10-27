#!/bin/bash
mkdir build
cd Thirdparty/DBoW2
mkdir build
cd build
cmake ..
make -j 4
cd ../../../build
make -j 4
