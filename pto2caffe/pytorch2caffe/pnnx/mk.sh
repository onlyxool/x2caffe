
if [ ! -d build ];then
    mkdir build
fi
cd build

cmake ..
make -j4
cp src/libpnnx.so ../../libpnnx.so
cd ..
