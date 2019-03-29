---
title: 'Ubuntu相关问题'
date: 2019-02-27 01:58:22
tags:
  -Ubuntu
---



Ubuntu ibus输入法无法按数字键取词问题解决方法。



参考：

[ubuntu自带输入法ibus 无法按数字键取词](https://blog.csdn.net/chen_minghui/article/details/80690821)



操作系统：Ubuntu 18.04



>**rm -rf ~/.cache/ibus/libpinyin** 
>有个人习惯词先备份user_bigram.db 
>重启 ibus restart

