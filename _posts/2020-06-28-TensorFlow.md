---
title: 'Tensorflow & CUDA'
date: 2020-06-28 14:35:00
tags:
  - 机器学习
  - 神经网络
  - tensorflow

---



pip环境和conda环境中TensorFlow及对应版本的CUDA和cuDNN的安装



**环境：**

操作系统：Ubunty 20.04

GPU：RTX2080Ti x2

python：3.7



### pip环境



##### CUDA:

- 在官网下载cuda_10.0.130_410.48_linux.run

- 在ubuntu20.04中，系统自带gcc版本高于CUDA要求的版本，安装时跳过版本检查即可

  - ```bash
    $ sudo bash cuda_10.0.130_410.48_linux.run --override
    ```

- 安装完成以后添加CUDA到环境变量

  - ```
    向 ~/.bashrc 中添加：
    	export  PATH=/usr/local/cuda-10.0/bin:$PATH
    	export  LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64$LD_LIBRARY_PATH
    
    然后在终端： source ~/.bashrc
    ```

- nvcc -V 可以查看CUDA版本，检查是否正确安装



##### cuDDN:

- 解压到当前目录

  - ```bash
    $ tar -xzvf cudnn-10.0-linux-x64-v7.6.5.32.tgz
    ```

- 复制cuDNN头文件

  - ```bash
    $ sudo cp cuda/include/* /usr/local/cuda-10.0/include/
    ```

- 复制cuDNN库

  - ```bash
    $ sudo cp cuda/lib64/* /usr/local/cuda-10.0/lib64/
    ```

- 添加可执行权限

  - ```bash
    $ sudo chmod +x /usr/local/cuda-10.0/include/cudnn.h
    $ sudo chmod +x /usr/local/cuda-10.0/lib64/libcudnn*
    ```

- 校验是否安装成功

  - ```bash
    $ cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
    ```

  - 成功则会出现如下信息：

    ```bash
    #define CUDNN_MAJOR 7
    #define CUDNN_MINOR 6
    #define CUDNN_PATCHLEVEL 5
    --
    #define CUDNN_VERSION (CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL)
    
    #include "driver_types.h"
    ```

    

##### tensorflow:

```bash
$ sudo pip3 install tensorflow-gpu==2.0.0beta0
```



### conda环境



```bash
$ conda install cudatoolkit=10.1
$ conda install cudnn=7.6
$ conda install tensorflow-gpu=2.1
```

