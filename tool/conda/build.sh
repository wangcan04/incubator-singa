export CMAKE_PREFIX_PATH=/root/miniconda2/

mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$PREFIX ..
make
make install
