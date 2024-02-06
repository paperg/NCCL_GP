#
# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#
# See LICENCE.txt for license information
#


.PHONY: all clean


TARGETS=src


build:
	g++ -g test_main.cpp -I../build/include  -I../fake_cuda/include -L../build/lib -L../fake_cuda/lib -lnccl -o test_main
	
clean:
	rm test_main 
