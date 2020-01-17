---
title: Shadowsocks（二）：客户端配置
date: 2019-02-23 00:48:22
tags:
  - Shadowsocks
---



Ubuntu环境下使用Shadowsocks科学上网。使用Vultr服务器搭建SS服务端，并在本地配置网络代理。







注：本机操作系统为 Ubuntu 18.04，服务器的搭建请参考上一篇文章。

[Shadowsocks（一）：基于Vultr搭建SS服务器](https://wyundi.github.io/2019/02/22/Shadowsocks%EF%BC%88%E4%B8%80%EF%BC%89%EF%BC%9A%E5%9F%BA%E4%BA%8EVultr%E6%90%AD%E5%BB%BASS%E6%9C%8D%E5%8A%A1%E5%99%A8/)





####  **参考：**



[Linux安装配置Shadowsocks客户端及开机自动启动](https://blog.huihut.com/2017/08/25/LinuxInstallConfigShadowsocksClient/)



[用SwitchyOmega管理代理设置](https://www.flyzy2005.com/tech/switchyomega-proxy-server/)



[Ubuntu终端使用Privoxy代理](https://www.codelast.com/%E5%8E%9F%E5%88%9B-ubuntu%E7%BB%88%E7%AB%AF%E4%BD%BF%E7%94%A8privoxy%E4%BB%A3%E7%90%86/)



#### **目录**：

> 1. Ubuntu环境下网络代理配置
>
>    * Shadowsocks无GUI客户端配置
>
>    * SHadowsocks_Qt5客户端配置
>
> 2. Chrome浏览器网络代理配置
>
> 3. 终端网络代理配置 







## **1. Ubuntu环境下网络代理配置**



###  Shadowsocks无GUI客户端配置





#### **安装**



安装Shadowsocks客户端需要python及其包管理工具pip，通过以下命令可以查看python和pip的版本：

```bash
$ python --version
$ pip --version
```



确定Python和pip都已经正确安装后，使用以下命令安装Shadowsocks客户端：

```bash
$ pip install shadowsocks
```





#### **配置**



创建Shadowsocks配置文件：

```bash
$ sudo touch /etc/shadowsocks/config.json
```



然后在该配置文件中添加服务器信息：

```bash
{

    "server":"my_server_ip",

    "server_port":my_server_port,

    "local_address": "127.0.0.1",

    "local_port":1080,

    "password":"my_password",

    "timeout":300,

    "method":"aes-256-cfb"
}
```



详细配置说明：

| Name          | 说明                       |
| ------------- | -------------------------- |
| Server        | 服务器地址，填IP地址或域名 |
| server_port   | 服务器开放端口             |
| local_address | 本地地址，127.0.0.1        |
| local_port    | 本地端口，一般为1080       |
| password      | 服务器密码                 |
| port_password | 服务器端口 + 密码          |
| timeout       | 超时重连                   |
| method        | 加密方式，默认aes-256-cfb  |
| fast_open     | TCP_FASTOPEN               |





#### **测试启动**

- 前端启动：sudo sslocal -c /etc/shadowsocks/config.json 
- 后台启动：sudo sslocal -c /etc/shadowsocks/config.json -d start
- 后台停止：sudo sslocal -c /etc/shadowsocks/config.json -d stop
- 重启：sudo sslocal -c /etc/shadowsocks/config.json -d restart





#### **开机启动**

使用Systemd来实现shadowsocks开机自启。

```bash
$ sudo vim /etc/systemd/system/shadowsocks.service
```



在里面填写如下内容：

```bash
[Unit]
Description=Shadowsocks Client Service
After=network.target

[Service]
Type=simple
User=root
ExecStart=/usr/bin/sslocal -c /etc/shadowsocks/config.json 

[Install]
WantedBy=multi-user.target
```



配置生效：

```bash
systemctl enable /etc/systemd/system/shadowsocks.service
```



输入管理员密码后配置生效。





### **Shadowsocks_Qt5客户端配置**





Qt5客户端使用snap应用商店的ss-qt。





#### **安装**

安装snap:

```bash
$ sudo apt update
$ sudo apt install snapd
```



安装ss-qt

```bash
sudo snap install ss-qt
```



![Alt text](/images/Shadowsocks_2/Screenshot from 2019-02-23 03-49-17.png)





#### **代理配置**



添加新的代理配置



![Alt text](/images/Shadowsocks_2/Screenshot from 2019-02-23 03-51-56.png)



保存后连接。





#### **开机启动**

在Startup Application中添加ss-qt即可。





## **2. Chrome浏览器网络代理配置**



#### **安装**



使用SwitchyOmege配置Chrome浏览器的网络代理。首先需要在Chrome应用商店内安装SwitchyOmege



在线安装直接打开Chrome应用商店，添加至Chrome即可。（需要科学上网）

离线安装需要先下载SwithcyOmega的离线安装包，下载地址：



> 1.GitHub：https://github.com/FelisCatus/SwitchyOmega/releases/latest
>
> 2.在线下载：https://www.switchyomega.com/download/



下载得到**SwitchyOmega_Chromium.crx**这个离线安装文件后，在Chrome地址栏输入**chrome://extensions**打开扩展程序，之后**打开开发者模式**，将离线安装文件拖入到Chrome中即可进行安装。





#### **配置代理**



安装成功后，先删除其自带的情景模式，然后点击New profile新建配置文件。



![Alt text](/images/Shadowsocks_2/Screenshot from 2019-02-23 10-44-32.png)



名称为Vultr，类型选择Proxy Profile，点击创建。



![Alt text](/images/Shadowsocks_2/Screenshot from 2019-02-23 11-04-29.png)



代理协议选择SOCKS5，服务器和端口填写本地IP地址和端口号（local_address和local_port）。

配置完成后点击Apply changes保存配置。



再新建一个配置文件，名称为AutoSwitch，类型选择Switch Profile。



![Alt text](/images/Shadowsocks_2/Screenshot from 2019-02-23 15-46-44.png)



选择 Add a rule list 添加规则列表。



![Alt text](/images/Shadowsocks_2/Screenshot from 2019-02-23 15-07-36.png)



Rule List Format 选择Autoproxy。

Rule List URL 填写：https://raw.githubusercontent.com/gfwlist/gfwlist/master/gfwlist.txt

这个地址是GFWList的地址，基本包含了常用的所有被墙网址，并且一直在更新。

选择 Download Profile Now 下载Profile。

然后将 Switch Rules 中对应的配置文件改为Vultr，保存配置即可。





## **3. 终端网络配置**



> 在Chrome上，是SwitchyOmega插件把HTTP和HTTPS流量转换成了socks协议的流量，才能使用socks代理。而Ubuntu终端是没有这样的协议转换的，所以没法直接使用sock5代理。这时候就需要一个协议转换器，例如Privoxy。





#### **安装**



```bash
$ sudo apt install privoxy
```





#### **配置**



修改privoxy配置文件/etc/privoxy/config，在文件末尾添加如下内容：

```bash
forward-socks5 / 127.0.0.1:1080 . # SOCKS5代理地址
listen-address 127.0.0.1:8080     # HTTP代理地址
forward 10.*.*.*/ .               # 内网地址不走代理
forward .abc.com/ .               # 指定域名不走代理
```



其中，第1行的 127.0.0.1:1080 是你在本地的SOCKS5代理地址，而第二行的 127.0.0.1:8080 则是SOCKS5转换成的 http 代理地址，最后两行指定了两个不走代理的地址。



配置好以后配置好之后重启Privoxy服务：

```bash
$ sudo /etc/init.d/privoxy restart
```



然后打开 /etc/profile，在最后添加以下两行：

```bash
export http_proxy="127.0.0.1:8080"
export https_proxy="127.0.0.1:8080"
```



即可在终端中科学上网。





#### **测试**



```bash
$ curl google.come
```



输入上述命令后显示Google首页的HTML代码即说明配置成功。

```bash
$ curl ip.gs
```



输入上述命令后显示当前IP地址。



由于执行ping指令使用ICMP传输协议，而SS代理是基于TCP或UDP协议，所以使用ping指令访问Google会超时。





