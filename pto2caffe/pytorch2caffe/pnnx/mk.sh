
if [ ! -d build ];then
#    rm build -rf
    mkdir build
fi
cd build

cmake ..
make -j4
cp src/libpnnx.so ../../libpnnx.so
cd ..
