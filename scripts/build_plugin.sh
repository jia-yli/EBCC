#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

SRC_DIR=${SCRIPT_DIR}/../src

opt=${1:-0}

case "$opt" in
  0)
    echo "Option is 0 - Simple Rebuild..."
    cd ${SRC_DIR}/build
    make && make install
    ;;
  1)
    echo "Option is 1 - Complete Rebuild..."
    rm -rf ${SRC_DIR}/build
    mkdir ${SRC_DIR}/build
    cd ${SRC_DIR}/build
    # cmake -DCMAKE_INSTALL_PREFIX=. -DCMAKE_BUILD_TYPE=Debug ..
    cmake -DCMAKE_INSTALL_PREFIX=. ..
    make && make install
    ;;
  *)
    echo "Unknown option: $opt"
    ;;
esac
