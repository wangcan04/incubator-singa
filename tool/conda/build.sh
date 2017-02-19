export CMAKE_INLCUDE_PATH=/Users/dbsystemnus/miniconda2/include
export CMAKE_LIBRARY_PATH=/Users/dbsystemnus/miniconda2/lib

mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$PREFIX ..
make
make install
