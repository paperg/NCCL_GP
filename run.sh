#!/bin/bash

make -j4 DEBUG=1 TRACE=1 NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"
