/*
 * @Author: Peng Guo & <<wyguopeng@163.com>
 * @Date: 2024-02-05 03:15:33
 * @LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @LastEditTime: 2024-02-06 23:58:20
 * @FilePath: /nccl-gp/src/graph/fake_cuda.cc
 * @Description: 
 * 
 * Copyright (c) 2024 by <wyguopeng@163.com, All Rights Reserved. 
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <sys/syscall.h>
#include <unistd.h>
#include "utils.h"
#include "topo.h"
#include "cuda_runtime.h"

#ifdef FAKE_CUDA_DUG
#define mlog(...) DebugLog(__func__, __LINE__, __VA_ARGS__)
#else
#define mlog(...)
#endif

void DebugLog( const char *filefunc, int line, const char *fmt, ...) {
    char buffer[1024];
    size_t len = 0;
    int pid = -1;
    int tid = -1;

    pid = getpid();
    tid = syscall(SYS_gettid);

    len = snprintf(buffer, sizeof(buffer), "%d:%d %s:%d | ", pid, tid, filefunc, line);

    if (len) {
        va_list vargs;
        va_start(vargs, fmt);
        len += vsnprintf(buffer+len, sizeof(buffer)-len, fmt, vargs);
        va_end(vargs);
        buffer[len++] = '\n';
        fwrite(buffer, 1, len, stdout);
    }
}

#define MAX_GPU 10

// GPU info
static struct{
    int64_t busid;
    unsigned int tid[32]; // 一个设备被 2个线程绑定
    int tid_num;
    int dev;
    int sm;
    int rank;
    int gdr;
} system_gpu[MAX_GPU] = {0};
static char exist_gpu_num = 0;

// 保留指针，后面用到一些信息
static struct ncclTopoSystem *local_sys_top;

int get_info_from_topo(struct ncclTopoSystem* system, int ngpu)
{
    int rank, i;
    local_sys_top = system;
    system->nodes[GPU].count = ngpu;
    exist_gpu_num = system->nodes[GPU].count;
    for(i = 0; i < exist_gpu_num; i++) {
        rank = system->nodes[GPU].nodes[i].gpu.rank;
        system_gpu[rank].busid  = system->nodes[GPU].nodes[i].id;
        system_gpu[rank].dev = system->nodes[GPU].nodes[i].gpu.dev;
        system_gpu[rank].sm = system->nodes[GPU].nodes[i].gpu.cudaCompCap;
        system_gpu[rank].rank = system->nodes[GPU].nodes[i].gpu.rank;
        system_gpu[rank].gdr = system->nodes[GPU].nodes[i].gpu.gdrSupport;
        mlog("Init from Topo, I can see GPU %d : id 0x%lx dev %d sm %d rank %d gdr %d\n", i, system_gpu[rank].busid, system_gpu[rank].dev,system_gpu[rank].sm,system_gpu[rank].rank,system_gpu[rank].gdr);
    }

    return 0;   
}

cudaError_t CUDARTAPI cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
{   
    mlog("%s : %s Line :%d", __FILE__, __func__, __LINE__);
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaStreamGetCaptureInfo(cudaStream_t stream, enum cudaStreamCaptureStatus *pCaptureStatus, unsigned long long *pId)
{
    mlog("%s : %s Line :%d", __FILE__, __func__, __LINE__);
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaHostRegister(void *ptr, size_t size, unsigned int flags)
{
    mlog("%s : %s Line :%d", __FILE__, __func__, __LINE__);
    return cudaSuccess;
}

__cudart_builtin__ cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream)
{
    mlog("%s : %s Line :%d", __FILE__, __func__, __LINE__);
    return cudaSuccess;
}

__cudart_builtin__ cudaError_t CUDARTAPI cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags)
{
    mlog("%s : %s Line :%d", __FILE__, __func__, __LINE__);
    return cudaSuccess;
}

__cudart_builtin__ cudaError_t CUDARTAPI cudaGetDeviceCount(int *count)
{
    mlog("%s : %s Line :%d", __FILE__, __func__, __LINE__);
    *count = 5;
    return cudaSuccess;
}

__cudart_builtin__ cudaError_t CUDARTAPI cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags)
{
    mlog("%s : %s Line :%d", __FILE__, __func__, __LINE__);
    return cudaSuccess;
}

__cudart_builtin__ cudaError_t CUDARTAPI cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream)
{
    mlog("%s : %s Line :%d", __FILE__, __func__, __LINE__);
    return cudaSuccess;
}

cudaError_t cudaLaunchKernel(const void *func, 
                                                            dim3 gridDim, dim3 blockDim, 
                                                            void **args, size_t sharedMem, 
                                                            cudaStream_t stream)
                                                            {
    mlog("%s : %s Line :%d", __FILE__, __func__, __LINE__);
    return cudaSuccess;
}

__cudart_builtin__ cudaError_t CUDARTAPI cudaFree(void *devPtr)
{
    if(devPtr)
        free(devPtr);
    mlog("%s : %s devPtr 0x%lx", __FILE__, __func__, devPtr);
    return cudaSuccess;
}

__cudart_builtin__ cudaError_t CUDARTAPI cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device)
{
    mlog("%s : %s Line :%d", __FILE__, __func__, __LINE__);
    return cudaSuccess;
}

__cudart_builtin__ cudaError_t CUDARTAPI cudaFuncSetAttribute(const void *func, enum cudaFuncAttribute attr, int value)
{
    mlog("%s : %s Line :%d", __FILE__, __func__, __LINE__);
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaGraphAddKernelNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaKernelNodeParams *pNodeParams)
{
    mlog("%s : %s Line :%d", __FILE__, __func__, __LINE__);
    return cudaSuccess;
}

__cudart_builtin__ cudaError_t CUDARTAPI cudaGetDevice(int *device)
{
    int tid = -1, i, j;
    tid = syscall(SYS_gettid);
    *device = 0;

    for(i = 0; i < exist_gpu_num; i++) {
        for(j = 0; j < system_gpu[i].tid_num; j++)
        {
            if(system_gpu[i].tid[j] == tid) {
                *device = system_gpu[i].dev;
            }
        }
      
    }

    mlog("%s : %s tid %d device %d", __FILE__, __func__, tid, *device);
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags)
{
    mlog("%s : %s Line :%d", __FILE__, __func__, __LINE__);
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaEventQuery(cudaEvent_t event)
{
    mlog("%s : %s Line :%d", __FILE__, __func__, __LINE__);
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaThreadExchangeStreamCaptureMode(enum cudaStreamCaptureMode *mode)
{
    mlog("%s : %s Line :%d", __FILE__, __func__, __LINE__);
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void *userData)
{
    mlog("%s : %s Line :%d", __FILE__, __func__, __LINE__);
    return cudaSuccess;
}

__cudart_builtin__ cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    mlog("%s : %s Line :%d", __FILE__, __func__, __LINE__);
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaEventCreate(cudaEvent_t *event)
{
    mlog("%s : %s Line :%d", __FILE__, __func__, __LINE__);
    return cudaSuccess;
}

__cudart_builtin__ cudaError_t CUDARTAPI cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device)
{
    // mlog("%s : %s Line :%d", __FILE__, __func__, __LINE__);
    return cudaSuccess;
}

__cudart_builtin__ cudaError_t CUDARTAPI cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *func)
{
    // mlog("%s : %s Line :%d", __FILE__, __func__, __LINE__);
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaDriverGetVersion(int *driverVersion)
{
    mlog("%s : %s Line :%d", __FILE__, __func__, __LINE__);
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaGraphAddHostNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, const struct cudaHostNodeParams *pNodeParams)
{
    mlog("%s : %s Line :%d", __FILE__, __func__, __LINE__);
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaDeviceGetPCIBusId(char *pciBusId, int len, int device)
{
    if (device > exist_gpu_num) {
        mlog("%s : %s Line_%d : device %d bigger than exist_gpu_num %d i know. Check !", __FILE__, __func__, __LINE__, device, exist_gpu_num);
    }
    int64ToBusId(system_gpu[device].busid, pciBusId);
    mlog("%s : %s device %d busId %s\n", __FILE__, __func__, device, pciBusId);
    
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaGraphAddEventWaitNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, cudaEvent_t event)
{
    mlog("%s : %s Line :%d", __FILE__, __func__, __LINE__);
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaIpcCloseMemHandle(void *devPtr)
{
    mlog("%s : %s Line :%d", __FILE__, __func__, __LINE__);
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaFreeHost(void *ptr)
{
    mlog("%s : %s Line :%d", __FILE__, __func__, __LINE__);
    free(ptr);
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaStreamGetCaptureInfo_v2(cudaStream_t stream, enum cudaStreamCaptureStatus *captureStatus_out, unsigned long long *id_out, cudaGraph_t *graph_out, const cudaGraphNode_t **dependencies_out, size_t *numDependencies_out)
{
    mlog("%s : %s Line :%d", __FILE__, __func__, __LINE__);
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaIpcOpenMemHandle(void **devPtr, cudaIpcMemHandle_t handle, unsigned int flags)
{
    mlog("%s : %s Line :%d", __FILE__, __func__, __LINE__);
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaDeviceSetLimit(enum cudaLimit limit, size_t value)
{
    mlog("%s : %s Line :%d", __FILE__, __func__, __LINE__);
    return cudaSuccess;
}

__cudart_builtin__ cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t event)
{
    mlog("%s : %s Line :%d", __FILE__, __func__, __LINE__);
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaGraphAddEventRecordNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph, const cudaGraphNode_t *pDependencies, size_t numDependencies, cudaEvent_t event)
{
    mlog("%s : %s Line :%d", __FILE__, __func__, __LINE__);
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaUserObjectCreate(cudaUserObject_t *object_out, void *ptr, cudaHostFn_t destroy, unsigned int initialRefcount, unsigned int flags)
{
    mlog("%s : %s Line :%d", __FILE__, __func__, __LINE__);
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaHostUnregister(void *ptr)
{
    mlog("%s : %s Line :%d", __FILE__, __func__, __LINE__);
    return cudaSuccess;
}

//device &  peerDevice 均为 busId
cudaError_t CUDARTAPI cudaDeviceCanAccessPeer(int *canAccessPeer, int device, int peerDevice)
{
    int link_cnt,i;
    struct ncclTopoNode* device_n = NULL;
    struct ncclTopoNode* peerDevice_n = NULL;
    *canAccessPeer = 0;

    device_n = &(local_sys_top->nodes[GPU].nodes[device]);
    peerDevice_n = &(local_sys_top->nodes[GPU].nodes[peerDevice]);

    link_cnt = device_n->nlinks;
    for(i = 0; i < link_cnt; i++) {
        if(device_n->links[i].remNode == peerDevice_n) {
            if(device_n->links[i].type == PATH_NVL) {
                *canAccessPeer = 1;
            }
            break;
        }
    }


    mlog("%s : %s device %d and %d P2P Cuda Check %d \n", __FILE__, __func__, device, peerDevice, *canAccessPeer);
    return cudaSuccess;
}

__cudart_builtin__ cudaError_t CUDARTAPI cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags)
{
    mlog("%s : %s Line :%d", __FILE__, __func__, __LINE__);
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags)
{
    mlog("%s : %s Line :%d", __FILE__, __func__, __LINE__);
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaSetDevice(int device)
{
    int tid = -1;
    tid = syscall(SYS_gettid);
    
    if(device > exist_gpu_num) {
        mlog("%s : %s device %d > Max GPU %d, Check", __FILE__, __func__, device, exist_gpu_num);
    }
    system_gpu[device].tid[system_gpu[device].tid_num++] = tid;
    system_gpu[device].dev = device;

    mlog("%s : %s tid %d bind device %d", __FILE__, __func__, tid, device);
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream)
{
    mlog("%s : %s Line :%d", __FILE__, __func__, __LINE__);
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaStreamUpdateCaptureDependencies(cudaStream_t stream, cudaGraphNode_t *dependencies, size_t numDependencies, unsigned int flags)
{
    mlog("%s : %s Line :%d", __FILE__, __func__, __LINE__);
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaPointerGetAttributes(struct cudaPointerAttributes *attributes, const void *ptr)
{
    mlog("%s : %s Line :%d", __FILE__, __func__, __LINE__);
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaHostAlloc(void **pHost, size_t size, unsigned int flags)
{
    mlog("%s : %s Line :%d", __FILE__, __func__, __LINE__);
    *pHost = malloc(size);
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaIpcGetMemHandle(cudaIpcMemHandle_t *handle, void *devPtr)
{
    mlog("%s : %s Line :%d", __FILE__, __func__, __LINE__);
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaGraphRetainUserObject(cudaGraph_t graph, cudaUserObject_t object, unsigned int count, unsigned int flags)
{
    mlog("%s : %s Line :%d", __FILE__, __func__, __LINE__);
    return cudaSuccess;
}

__cudart_builtin__ cudaError_t CUDARTAPI cudaMalloc(void **devPtr, size_t size)
{
    *devPtr = malloc(size);
    mlog("%s : %s devPrt 0x%lx size %d", __FILE__, __func__, *devPtr, size);
    return cudaSuccess;
}

cudaError_t CUDARTAPI cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
    mlog("%s : %s Line :%d", __FILE__, __func__, __LINE__);
    return cudaSuccess;
}

__cudart_builtin__ cudaError_t CUDARTAPI cudaGetLastError(void)
{
    mlog("%s : %s Line :%d", __FILE__, __func__, __LINE__);
    return cudaSuccess;
}

__cudart_builtin__ const char* CUDARTAPI cudaGetErrorString(cudaError_t error)
{
    mlog("%s : %s Line :%d", __FILE__, __func__, __LINE__);
    return "Fake Error message";
}