#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

SRC_DIR=${SCRIPT_DIR}/../src

rm -rf ${SRC_DIR}/build
mkdir ${SRC_DIR}/build
cd ${SRC_DIR}/build
# cmake -DCMAKE_INSTALL_PREFIX=. -DCMAKE_BUILD_TYPE=Debug ..
cmake -DCMAKE_INSTALL_PREFIX=. ..
make && make install
