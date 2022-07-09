cd ~/cudaPMF
sudo mkdir -p /task6_result/spirt
mkdir -p build
cd build
rm -r *
cmake ..
make -j4