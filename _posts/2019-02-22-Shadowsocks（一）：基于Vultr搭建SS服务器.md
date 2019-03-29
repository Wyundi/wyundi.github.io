---
title: Shadowsocks（一）：基于Vultr搭建SS服务器
date: 2019-02-22 19:26:39
tags:
  -Shadowsocks
---



Ubuntu环境下使用Shadowsocks科学上网。使用Vultr服务器搭建SS服务端，并在本地配置网络代理。



# **Shadowsocks（一）：基于Vultr搭建SS服务器**





注：本机操作系统为 Ubuntu 18.04， 图中所示服务器仅为测试用例，将在结束后销毁。


## **Vultr服务器**

Vultr是美国的一个VPS服务商，全球有15个数据中心，可以一键部署服务器。采用小时计费策略，可以在任何时间新建或者摧毁VPS服务器。价格低廉，最便宜的只要2.5一个月，支持支付宝及微信支付。

Vultr官网连接：[Vultr The Infrastructure Cloud™](https://www.vultr.com/)


### **新用户注册**

![Alt text](/images/Shadowsocks_1/Screenshot from 2019-02-22 14-19-26.png)

打开Vultr官网，在相应的位置填写邮箱和密码，然后创建账户即可。注册完会受到一封验证邮件，点击连接确认注册。


### **账户充值**

Vultr的服务器价格从一个月2.5美元到640美元不等，可根据个人需求选择相应的服务器方案。对服务器需求较小的个人用户可选择一个月5美元的计费方案，其中包含了25GB的SSD，1个CPU，1024MB内存以及1000GB的流量。Vultr的收费方式实际上是按小时计费的，比如5美元一个月的服务器对应的收费标准为0.07美元一小时。记时从开通服务器开始，直到服务器被销毁为止。费用会自动从账户中扣除。在Billing页面可以查看当前余额并进行充值，支持的支付方式包括信用卡、Paypal、支付宝以及微信等，还可以用Gift Code充值。

![Alt text](/images/Shadowsocks_1/Screenshot from 2019-02-22 14-51-54.png)


### **创建服务器**

充值成功后，进入Servers页面，点击右上角的加号创建服务器。

![Alt text](/images/Shadowsocks_1/Screenshot from 2019-02-22 15-02-24.png)

选择服务器配置，包括服务器位置，操作系统，机器配置以及附加服务等。

![Alt text](/images/Shadowsocks_1/Screenshot from 2019-02-22 15-06-05.png)

![Alt text](/images/Shadowsocks_1/Screenshot from 2019-02-22 15-04-08.png)

![Alt text](/images/Shadowsocks_1/Screenshot from 2019-02-22 15-07-37.png)

配置完成后点击Deploy Now完成创建。


### **配置服务器**

服务器创建成功后，你就可以从Servers页面看到你的服务器。待其安装完成后，点击服务器可以查看其详细信息。其中包含你的服务器的IP地址，用户名和密码。

![Alt text](/images/Shadowsocks_1/Screenshot from 2019-02-22 15-15-37.png)

通过Linux终端连接VPS。

```bash
$ ssh -l 远程服务器用户名 服务器ip地址
```

随后输入密码即可连接远程服务器。

![Alt text](/images/Shadowsocks_1/Screenshot from 2019-02-22 18-54-28.png)

打开服务器终端以后开始配置和Shadowsocks， 这里采用[teddysun](https://teddysun.com/342.html)的一键安装脚本。
逐行输入以下命令：

```bash
$ wget --no-check-certificate https://raw.githubusercontent.com/teddysun/shadowsocks_install/master/shadowsocks.sh
$ chmod +x shadowsocks.sh
$ ./shadowsocks.sh 2>&1 | tee shadowsocks.log
```

程序运行后将会出现如下所示界面：

![Alt text](/images/Shadowsocks_1/Screenshot from 2019-02-22 19-02-05.png)

为你的Shadowsocks服务器设置密码。

![Alt text](/images/Shadowsocks_1/Screenshot from 2019-02-22 19-05-59.png)

设置端口。

![Alt text](/images/Shadowsocks_1/Screenshot from 2019-02-22 19-06-32.png)

选择加密方式。

![Alt text](/images/Shadowsocks_1/Screenshot from 2019-02-22 19-07-02.png)

设置完成，按任意键开始配置Shadowsocks服务器。

![Alt text](/images/Shadowsocks_1/Screenshot from 2019-02-22 19-11-57.png)

配置成功，终端将显示本机的Shadowsocks配置信息，将其保存后可用于配置客户端的Shadowsocks连接。