mkdir psa_mod/build/
mkdir ccvt_mod/build/
mkdir dyadic_mod/build/
cd psa_mod/build/
cmake ../CMakeLists.txt
make
cd ../../ccvt_mod/build/
cmake ../CMakeLists.txt
make
cd ../../dyadic_mod/build/
cmake ../CMakeLists.txt
make
cd ../..
