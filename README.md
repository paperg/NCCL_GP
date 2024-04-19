
[toc]

# 程序运行编译流程

为了便于学习 NCCL 源码，但是实际环境设备组成又不可能多样化，毕竟无论是GPU 还是 NVLINK 都需要一定的成本，所以，修改 nccl 源码，使其脱离硬件，可以在没有硬件支持的基础上进行软件调试。

基于 NCCL 2.19.1

主要思路是：
1. 手动修改设计网络拓扑图，与 NCCL 源码实际硬件连接生成的 xml 文件要一样；
2. 测试程序启动后，会读取 xml 文件，通过原 API 接口 ncclTopoGetSystem() 解析 xml 文件，原代码中的生成xml 文件已经取消；
3. 部分涉及 GPU 的接口什么事也不做，或者注释，请比对原代码。

## 环境变量设置

**！！！ 环境变量必须设置，否则编译一定会有问题 ！！！**
***只有 NCCL_ROOT_DIR 需要自定义，其他的不变即可***
***修改 NCCL_TOPO_FILE 变量指定拓扑XML文件，可以修改文件，设计拓扑结构, 默认使用 topo/nvlink_5GPU.xml***
1. NCCL_ROOT_DIR
NCCL_ROOT_DIR 指向源码路径，最后的根号不需要

2. CUDA_LIB & CUDA_INC 
   依赖的 CUDA 库，因为是虚拟的，自己写的函数，只是调用返回，所以头文件与动态库要指定 fake_cuda 目录下路径

3. LD_LIBRARY_PATH
    编译时需要的动态库路径

4. NCCL_TOPO_FILE
   运行需要的拓扑 xml 文件，需要手动修改

5. NCCL_GRAPH_DUMP_FILE
   生成的图

6. GPU_DEV_NUM
   指定网络图谱中 GPU 的设备数量，测试文件需要，默认为 3 个

7. NCCL_DEBUG & NCCL_DEBUG_SUBSYS 
   NCCL 自带的调试日志级别，级别最大，更多的打印

```bash

export NCCL_ROOT_DIR=/home/gp/NCCL_GP
export CUDA_LIB=$NCCL_ROOT_DIR/fake_cuda/lib
export CUDA_INC=$NCCL_ROOT_DIR/fake_cuda/include
export LD_LIBRARY_PATH=$NCCL_ROOT_DIR/fake_cuda/lib:$NCCL_ROOT_DIR/build/lib
export NCCL_TOPO_FILE=$NCCL_ROOT_DIR/topo/nvlink_5GPU.xml
export NCCL_GRAPH_DUMP_FILE=$NCCL_ROOT_DIR/topo/graph_dump.xml
export GPU_DEV_NUM=5
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=ALL

```

## 编译命令

**Note:** 编译前确保 run.sh 以及  src/collectivates/device/gen_rules.sh 两个脚本有可执行权限，使用如下命令增加可执行权限。

```bash
chmod +x ./run.sh
chmod +x src/collectivates/device/gen_rules.sh
```

1. 直接运行脚本
```bash
./run.sh
```

1. 或者使用命令
```bash
make -j4 DEBUG=1 TRACE=1 VERBOSE=1 NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"
```

## 测试
测试程序，或者说程序入口在 test/ 目录下，编译完成后会有 test_main 可执行程序。
test/test_dev_process.cpp 是使用 MPI 的多进程测试程序。

# 其他

在路径topo/ 下提供了一个 xml 转 图片的 python3 工具：xml_to_PNG.py
* 依赖库
    python3
    graphviz : 需要安装,  pip install graphviz

* 命令：
```bash
python3 xml_to_PNG.py -i input_xml_file -o output_file_name
```
