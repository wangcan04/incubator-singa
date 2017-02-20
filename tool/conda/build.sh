export CMAKE_PREFIX_PATH=/root/miniconda2/

echo $PREFIX
echo $PATH

mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$PREFIX ..
make
make install
