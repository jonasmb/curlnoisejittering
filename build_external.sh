mkdir psa_mod/build/
mkdir ccvt_mod/build/
mkdir dyadic_mod/build/
cmake -Hpsa_mod -Bpsa_mod/build/ -DCMAKE_BUILD_TYPE=Release
cmake --build psa_mod/build --config Release
cmake -Hccvt_mod -Bccvt_mod/build/ -DCMAKE_BUILD_TYPE=Release
cmake --build ccvt_mod/build --config Release
cmake -Hdyadic_mod -Bdyadic_mod/build/ -DCMAKE_BUILD_TYPE=Release
cmake --build dyadic_mod/build --config Release
