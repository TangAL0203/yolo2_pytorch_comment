# -*- coding:utf-8 -*-
import torch
from utils import convert2cpu

def parse_cfg(cfgfile):
    blocks = [] # 列表里面存放字典
    fp = open(cfgfile, 'r')
    block =  None
    line = fp.readline() # 每次读取一行
    while line != '':
        line = line.rstrip() # str.rstrip([chars]) 删除 string 字符串末尾的指定字符（默认为空格）
        if line == '' or line[0] == '#':  # 遇到#忽略
            line = fp.readline()
            continue        
        elif line[0] == '[': # []包含的为一个模块
            if block:
                blocks.append(block) # block存放当前获取的信息
            block = dict() # 第一次创建一个空字典
            block['type'] = line.lstrip('[').rstrip(']') # 存放[]包含的模块的类型，比如convolutional
            # set default value 默认卷积层，batch_normalize不使用
            if block['type'] == 'convolutional': 
                block['batch_normalize'] = 0
        else:
            key,value = line.split('=')
            key = key.strip() # 移除字符串头尾指定的字符（默认为空格）
            if key == 'type':
                key = '_type'
            value = value.strip()
            block[key] = value
        line = fp.readline()

    if block:
        blocks.append(block) # 解析最后一行参数
    fp.close()
    return blocks

## 打印出cfg文件中，各层的参数信息
## 层名 滤波器个数 滤波器尺寸 输入 输出
def print_cfg(blocks):
    print('layer     filters    size              input                output');
    prev_width = 416  # YOLO2预定义尺寸
    prev_height = 416
    prev_filters = 3
    out_filters =[]
    out_widths =[]
    out_heights =[]
    ind = -2 # net块为-1,第一个卷积层序号为0
    for block in blocks:  # block是字典，存放层的参数信息
        ind = ind + 1
        if block['type'] == 'net':
            prev_width = int(block['width']) # 获取net块中的prev_width信息
            prev_height = int(block['height']) # 处理之前，即输入
            continue
        elif block['type'] == 'convolutional':
            filters = int(block['filters'])
            kernel_size = int(block['size'])
            stride = int(block['stride'])
            is_pad = int(block['pad'])
            pad = (kernel_size-1)/2 if is_pad else 0 # 计算填充的像素个数，两边各填充一个pad
            width = (prev_width + 2*pad - kernel_size)/stride + 1 # width，height 为 输出
            height = (prev_height + 2*pad - kernel_size)/stride + 1
            print('%5d %-6s %4d  %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (ind, 'conv', filters, kernel_size, kernel_size, stride, prev_width, prev_height, prev_filters, width, height, filters))
            prev_width = width # 更新prev_width，作为下一层的输入
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'maxpool':
            pool_size = int(block['size']) # 做max pooling的输入都是偶数
            stride = int(block['stride'])
            width = prev_width/stride
            height = prev_height/stride
            print('%5d %-6s       %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (ind, 'max', pool_size, pool_size, stride, prev_width, prev_height, prev_filters, width, height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'avgpool':
            width = 1 # yolo2中没有avgpool
            height = 1 # avgpool输出为1x1，全局平均池化
            print('%5d %-6s                   %3d x %3d x%4d   ->  %3d' % (ind, 'avg', prev_width, prev_height, prev_filters,  prev_filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'softmax':   # 分类任务有softmax层
            print('%5d %-6s                                    ->  %3d' % (ind, 'softmax', prev_filters))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'cost':
            print('%5d %-6s                                     ->  %3d' % (ind, 'cost', prev_filters))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'reorg':  # reorg层，match前后不同层的特征图尺寸，做拼接
            stride = int(block['stride'])
            filters = stride * stride * prev_filters # 新的输出为之前的stride * stride倍
            width = prev_width/stride # 比如上一层是26x26，那么输出应该是13x13
            height = prev_height/stride
            print('%5d %-6s             / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (ind, 'reorg', stride, prev_width, prev_height, prev_filters, width, height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'route': # ind为当前的层的序号
            layers = block['layers'].split(',') # layers=-9 或 layers=-1,-4（-9因该是以该层为基准，往后倒退9层）
            layers = [int(i) if int(i) > 0 else int(i)+ind for i in layers] # 获取相关层的绝对序号
            if len(layers) == 1:
                print('%5d %-6s %d' % (ind, 'route', layers[0]))
                prev_width = out_widths[layers[0]] # 输入是layers[0]层，比如layers=-9，将第16层输出作为该层的输入
                prev_height = out_heights[layers[0]]
                prev_filters = out_filters[layers[0]]
            elif len(layers) == 2:
                print('%5d %-6s %d %d' % (ind, 'route', layers[0], layers[1]))
                prev_width = out_widths[layers[0]]
                prev_height = out_heights[layers[0]]
                assert(prev_width == out_widths[layers[1]]) # 要求拼接的两层feature map尺寸一致
                assert(prev_height == out_heights[layers[1]])
                prev_filters = out_filters[layers[0]] + out_filters[layers[1]] # 当前层的输出通道数为两者之和
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'region':  # 1x1卷积之后输出预测值，在region层计算损失
            print('%5d %-6s' % (ind, 'detection'))
            out_widths.append(prev_width) # 
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'shortcut': #两个层短接，跳跃地连接两层。将from层的输出转到该层
            from_id = int(block['from'])
            from_id = from_id if from_id > 0 else from_id+ind
            print('%5d %-6s %d' % (ind, 'shortcut', from_id))
            prev_width = out_widths[from_id]
            prev_height = out_heights[from_id]
            prev_filters = out_filters[from_id]
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'connected': # 全连接层，输出1x1x类别数
            filters = int(block['output'])
            print('%5d %-6s                            %d  ->  %3d' % (ind, 'connected', prev_filters,  filters))
            prev_filters = filters
            out_widths.append(1)
            out_heights.append(1)
            out_filters.append(prev_filters)
        else:
            print('unknown type %s' % (block['type']))

# 输入参数
# conv_model：网络模型； buf：存放初始化参数的numpy数组；start：读取Numpy数组的开始序号
# 功能：从buf读取数据，对conv_model进行初始化
def load_conv(buf, start, conv_model):
    num_w = conv_model.weight.numel() # numel()返回Tensor中元素的个数
    num_b = conv_model.bias.numel()
    # bias.data和weight.data都是tensor，将buf中的数据拷贝到weight.data和bias.data
    conv_model.bias.data.copy_(torch.from_numpy(buf[start:start+num_b]));   start = start + num_b
    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_w])); start = start + num_w
    return start

# 读取conv_model中卷积层的参数并保存为numpy数组文件（包括bias和weight）
def save_conv(fp, conv_model):
    if conv_model.bias.is_cuda:
        convert2cpu(conv_model.bias.data).numpy().tofile(fp) # tofile()将array数组写到一个文件中，以文本或则二进制(默认)的形式。
        convert2cpu(conv_model.weight.data).numpy().tofile(fp)
    else:
        conv_model.bias.data.numpy().tofile(fp)
        conv_model.weight.data.numpy().tofile(fp)

# 函数功能
# 
def load_conv_bn(buf, start, conv_model, bn_model):
    num_w = conv_model.weight.numel()
    num_b = bn_model.bias.numel()
    bn_model.bias.data.copy_(torch.from_numpy(buf[start:start+num_b]));     start = start + num_b
    bn_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_b]));   start = start + num_b
    bn_model.running_mean.copy_(torch.from_numpy(buf[start:start+num_b]));  start = start + num_b
    bn_model.running_var.copy_(torch.from_numpy(buf[start:start+num_b]));   start = start + num_b
    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_w])); start = start + num_w 
    return start

def save_conv_bn(fp, conv_model, bn_model):
    if bn_model.bias.is_cuda:
        convert2cpu(bn_model.bias.data).numpy().tofile(fp)
        convert2cpu(bn_model.weight.data).numpy().tofile(fp)
        convert2cpu(bn_model.running_mean).numpy().tofile(fp)
        convert2cpu(bn_model.running_var).numpy().tofile(fp)
        convert2cpu(conv_model.weight.data).numpy().tofile(fp)
    else:
        bn_model.bias.data.numpy().tofile(fp)
        bn_model.weight.data.numpy().tofile(fp)
        bn_model.running_mean.numpy().tofile(fp)
        bn_model.running_var.numpy().tofile(fp)
        conv_model.weight.data.numpy().tofile(fp)

# 读取buf值，对fc_model的weight和bias初始化
def load_fc(buf, start, fc_model):
    num_w = fc_model.weight.numel()
    num_b = fc_model.bias.numel()
    fc_model.bias.data.copy_(torch.from_numpy(buf[start:start+num_b]));
    start = start + num_b
    fc_model.weight.data.copy_(torch.from_numpy(buf[start:start+num_w]));
    start = start + num_w 
    return start

# 将fc_model值保存为numpy数组文件
def save_fc(fp, fc_model):
    fc_model.bias.data.numpy().tofile(fp)
    fc_model.weight.data.numpy().tofile(fp)

if __name__ == '__main__':
    import sys
    blocks = parse_cfg('cfg/yolo.cfg')
    if len(sys.argv) == 2:
        blocks = parse_cfg(sys.argv[1])
    print_cfg(blocks)
