---
title: 'Tensorflow & CUDA'
date: 2020-01-09 20:00:00
tags:
  - 机器学习
  - 神经网络
  - tensorflow
---





TensorFlow及对应的CUDA版本问题



# Tensorflow & CUDA



**参考：**

[CUDA 440.33.01版驱动文档](https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-440-3301/index.html)

[TensorFlow安装及版本问题](https://www.tensorflow.org/install/source#common_installation_problems)



**环境：**

操作系统：Ubuntu 18.04

GPU：GTX 960M



### 问题描述



将TensorFlow版本从1.2升级到了2.1以后运行报错。提示缺少libnvinfer.so。

```bash
$ " Could not load dynamic library 'libnvinfer.so.6' "
```



产生这个错误的原因是CUDA版本和TensorFlow不匹配。



### 解决方案



将原有的TensorFlow卸载

```bash
$ sudo pip uninstall protobuf
$ sudo pip uninstall tensorflow
```



从[NVIDIA官网](https://developer.nvidia.com/cuda-toolkit-archive)下载对应GPU版本的CUDA。CUDA10.2在960M环境中无法使用，所以下载[10.0版本](https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=runfilelocal)，然后根据说明进行安装。

> Installation Instructions:
>
> 	1. Run `sudo sh cuda_10.0.130_410.48_linux.run`
>  	2. Follow the command-line prompts



查看对应CUDA 10.0 的TensorFlow版本，最高为2.0.0。安装该版本的TensorFlow。

```bash
$ sudo pip install tensorflow==2.0.0
```



### 查看CUDA和TensorFlow版本



CUDA：

```bash
$ cat /usr/local/cuda/version.txt
```



TensorFlow:

```python
$ python3
>>> import tensorflow as tf
>>> tf.__version__
'2.0.0'
```

