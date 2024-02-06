/*
 * @Author: Peng Guo & <wyguopeng@163.com>
 * @Date: 2024-02-05 17:32:12
 * @LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @LastEditTime: 2024-02-05 19:13:04
 * @FilePath: /nccl-gp/src/collectives/device/fake_device_code.c
 * @Description: 
 * 
 * Copyright (c) 2024 by wyguopeng@163.com, All Rights Reserved. 
 */
#include "common.h"

IMPL_COLL_R(AllReduce);
IMPL_COLL_R(ReduceScatter);
IMPL_COLL_R(Reduce);
IMPL_COLL_C(Broadcast);
IMPL_COLL_C(AllGather);
IMPL_COLL_P(SendRecv);