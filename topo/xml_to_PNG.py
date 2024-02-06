
"""
 * @Author: Peng Guo & <wyguopeng@163.com>
 * @Date: 2024-02-05 02:52:31
 * @LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @LastEditTime: 2024-02-06 21:19:27
 * @FilePath: /nccl-gp/README_GP.md
 * @Description: 
 * 
 * Copyright (c) 2024 by engguopeng@gmail.com, All Rights Reserved. 
"""

import xml.etree.ElementTree as ET
from graphviz import Graph
import os
import sys
import getopt

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

def build_topology(graph, parent_element, parent_node=None):
    dev = parent_element.tag
    res = None
    if dev == 'cpu':
        res = parent_element.get('numaid').replace(':', '_')
    elif dev == 'pci':
        res = parent_element.get('busid').replace(':', '_')
    elif dev == 'gpu':
        res = parent_element.get('dev').replace(':', '_')
    elif dev == 'nvlink':
        res = parent_element.get('target').replace(':', '_')
    elif dev == 'net':
        res = parent_element.get('name').replace(':', '_')
    
    node = dev
    if res is not None:
        node = dev + '_' + res
    
    all_attributes = {attr: parent_element.get(attr) for attr in parent_element.attrib}
    str_attr = node + '\n\n' + str(all_attributes).replace(',', '\n')
    graph.node(node, shape='box', label=str_attr)
    
    if parent_node is not None:
        graph.edge(parent_node, node)
    
    for child_element in parent_element:
        build_topology(graph, child_element, node)

def usage():
    print('Usage: python3 xml_to_PNG.py -i input_file_name -o out_file_name')
    
  

if len(sys.argv) != 5:
    usage()
    sys.exit()
    
argv = sys.argv[1:]

opts, args = getopt.getopt(argv, "hi:o:")  # 短选项模式
for opt, arg in opts:
    if opt == '-i':
        input_file = arg
    elif opt == '-o':
        output_file = arg
    elif opt == '-h': 
        usage()
        sys.exit()
    
print(f'Parse topology file {output_file} to PDF or picture file {output_file}')

# 解析XML数据
tree = ET.parse(input_file)
# 获取 XML 文档对象的根结点 Element
root_element = tree.getroot()

graph = Graph()
build_topology(graph, root_element)

# 绘制拓扑图
graph.render(output_file, format='png')
graph.view()
print('Finished')