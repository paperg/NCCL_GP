/*
 * @Author: Peng Guo & <wyguopeng@163.com>
 * @Date: 2024-01-24 00:37:34
 * @LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @LastEditTime: 2024-04-02 11:02:29
 * @FilePath: /nccl-gp/test/test_main.cpp
 * @Description: 
 * 
 * Copyright (c) 2024 by wyguopeng@163.com, All Rights Reserved. 
 */
#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "nccl.h"

#define CUDACHECK(cmd) do {                         \
  cudaError_t err = cmd;                            \
  if (err != cudaSuccess) {                         \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(err)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

int main(int argc, char* argv[])
{
    ncclComm_t comms[4];
    int nDev = 3;

    char* env = getenv("GPU_DEV_NUM");
    if (env) {
      nDev = atoi(env);
      printf("GPU Dev num set by User %d\n", nDev);
    }
    //managing 4 devices
    int i;
    int size = 4*1024;
    int devs[32] = { 0, 1, 2};

    for(i = 0; i < nDev; i++) {
      devs[i] = i;
    }

    // allocating and initializing device buffers
    float** sendbuff = (float**)malloc(nDev * sizeof(float*));
    float** recvbuff = (float**)malloc(nDev * sizeof(float*));
    cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);

    //initializing NCCL
    NCCLCHECK(ncclCommInitAll(comms, nDev, devs));

    //calling NCCL communication API. Group API is required when using
    // multiple devices per thread
    // NCCLCHECK(ncclGroupStart());
    // for (int i = 0; i < nDev; ++i)
    //     NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum,
    //         comms[i], s[i]));
    // NCCLCHECK(ncclGroupEnd());

    // free device buffers
    for (int i = 0; i < nDev; ++i) {
        free(sendbuff[i]);
        free(recvbuff[i]);
    }

    if(s)
      free(s);

    //finalizing NCCL
    for(int i = 0; i < nDev; ++i)
        if(comms[i])
            ncclCommDestroy(comms[i]);


    printf("Success \n");

    return 0;
}