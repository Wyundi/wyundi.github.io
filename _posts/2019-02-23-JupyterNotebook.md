---
title: 'Jupyter notebook'
date: 2019-02-23 17:10:53
tags:
  -Jupyter Notebook
---



Jupyter Notebook 相关问题。



## Jupyter notebook Error: ''create_prompt_application''



将prompt-toolkit版本降至1.0.15即可

```bash
$ sudo pip install 'prompt-toolkit==1.0.15'
```



## 后台运行



创建文件 jupyter.sh，向其中添加如下代码：

```python
nohup jupyter notebook > jupyter.log 2>&1 &
```



保存后向文件加入可执行权限：

```bash
$ chmod +x jupyter.sh
```



下次需要启动 jupyter notebook 时运行 jupyter.sh 即可，日志文件将会保存在 jupyter.log 中。