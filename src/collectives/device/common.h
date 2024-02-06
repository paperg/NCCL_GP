

#ifndef NCCL_DEVICE_COMMON_H_
#define NCCL_DEVICE_COMMON_H_

#include <stdio.h>
#include <stdint.h>

#define NCCL_FUNC_NAME(func, algo, proto, devredop, type) \
  ncclFunction_##func##_##algo##_##proto##_##devredop##_##type

// Examples :     AllReduce, RING, LL,    Sum,   uint8
#define IMPL_COLL_FUNC(func, algo, proto, devredop, type) \
  void NCCL_FUNC_NAME(func, algo, proto, devredop, type)() { \
    printf("Func %s\n", __func__); \
}

#define NCCL_KERN_NAME(func, algo, proto, devredop, type) \
  ncclKernel_##func##_##algo##_##proto##_##devredop##_##type

// Only generate kernels for SUM
#if NCCL_OP == 0
#define IMPL_COLL_KERN(func, algo, proto, devredop, type, fIndex) \
  void NCCL_KERN_NAME(func, algo, proto, devredop, type)( \
    struct ncclDevComm* comm, uint64_t channelMask, struct ncclWork* workHead \
  ) { \
    printf("%s\n", __func__); \
}
#else
#define IMPL_COLL_KERN(func, algo, proto, devredop, type, fInded)
#endif

// Only generate inline kernels for LL
#define IMPL_COLL4(func, algo, devredop, type, ncclType) \
  IMPL_COLL_FUNC(func, algo, LL,     devredop, type) \
  IMPL_COLL_FUNC(func, algo, LL128,  devredop, type) \
  IMPL_COLL_FUNC(func, algo, SIMPLE, devredop, type) \
  IMPL_COLL_KERN(func, algo, LL,     devredop, type, 0)

#define IMPL_COLL3(func, devredop, type, ncclType) \
  IMPL_COLL4(func, TREE,    devredop, type, ncclType) \
  IMPL_COLL4(func, RING,    devredop, type, ncclType) \
  IMPL_COLL4(func, COLLNET_DIRECT, devredop, type, ncclType) \
  IMPL_COLL4(func, COLLNET_CHAIN, devredop, type, ncclType) \
  IMPL_COLL4(func, NVLS, devredop, type, ncclType) \
  IMPL_COLL4(func, NVLS_TREE, devredop, type, ncclType)

#if NCCL_TYPE == 0
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, int8_t,   ncclInt8)
#elif NCCL_TYPE == 1
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, uint8_t,  ncclUint8)
#elif NCCL_TYPE == 2
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, int32_t,  ncclInt32)
#elif NCCL_TYPE == 3
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, uint32_t, ncclUint32)
#elif NCCL_TYPE == 4
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, int64_t,  ncclInt64)
#elif NCCL_TYPE == 5
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, uint64_t, ncclUint64)
#elif NCCL_TYPE == 6
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, half,     ncclFloat16)
#elif NCCL_TYPE == 7
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, float,    ncclFloat32)
#elif NCCL_TYPE == 8
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, double,   ncclFloat64)
#elif NCCL_TYPE == 9
#define IMPL_COLL2(func, devredop) IMPL_COLL3(func, devredop, __nv_bfloat16, ncclBfloat16)
#endif

// Reduction define all functions
#if NCCL_OP == 0
#define IMPL_COLL_R(func) IMPL_COLL2(func, Sum);
#elif NCCL_OP == 1
#define IMPL_COLL_R(func) IMPL_COLL2(func, Prod);
#elif NCCL_OP == 2
#define IMPL_COLL_R(func) IMPL_COLL2(func, Min);
#elif NCCL_OP == 3
#define IMPL_COLL_R(func) IMPL_COLL2(func, Max);
#elif NCCL_OP == 4
#define IMPL_COLL_R(func) IMPL_COLL2(func, PreMulSum);
#elif NCCL_OP == 5
  #if NCCL_TYPE < 6
    #define IMPL_COLL_R(func) IMPL_COLL2(func, SumPostDiv);
  #else
    #define IMPL_COLL_R(func) // skip SumPostDiv for floating point
  #endif
#endif

#if NCCL_OP == 0 && NCCL_TYPE == 0
// Copy primitives only define one function for copy
#define IMPL_COLL_C(func) IMPL_COLL3(func, Sum, int8_t, ncclInt8);

// Point-to-point primitives only have one function/kernel.
#define IMPL_COLL_P(func) \
  IMPL_COLL_FUNC(func, RING, SIMPLE, Sum, int8_t); \
  IMPL_COLL_KERN(func, RING, SIMPLE, Sum, int8_t, 0);
#else
#define IMPL_COLL_C(func)
#define IMPL_COLL_P(func)
#endif

#endif
