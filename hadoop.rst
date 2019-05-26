hadoop
================

Hadoop 是一个由 Apache 基金会所开发的分布式大数据存储和处理架构。

它实现了一个分布式文件系统（Hadoop Distributed File System），简称 HDFS，用于大数据的存储以及 MapReduce 机制对大数据高效处理。

Hadoop 具有：

- 高容错性：不依赖于底层硬件，在软件层面维护多个数据副本，确保出现失败后针对这类节点重新进行分布式处理；
- 可以部署在大量的低廉硬件上；
- 针对离线大数据处理的并行处理框架：MapReduce；
- 流式文件访问，一次性写入，多次读取，保证数据一致性。

安装和配置
-------------

版本选择
~~~~~~~~~~~~

在众多的 Linux 发行版中，Ubuntu 通常作为桌面系统使用，具有漂亮的用户界面和高度的软件安装便利性。

与桌面系统不同，服务器要求高性能和高可靠性，所以通常使用 CentOS，Debian 或者 SuSe。

CentOS （Community Enterprise Operating System，社区企业操作系统）。它基于 RHEL （Red Hat Enterprise Linux）依 照开放源代码规定释出的源代码所编译而成。由于出自同样的源代码，因此有些要求高度稳定性的服务器以 CentOS 替代商业版的 RHEL 使用。两者的不同在于CentOS完全开源。

CentOS/RHEL 版本的生命周期具有 7-10 年之久，基本上可以覆盖硬件的生命周期，在整个周期中，软件漏洞都会得到及时的安全补丁支持。这里使用
`CentOS-7-x86_64-DVD-1810.iso <http://isoredirect.centos.org/centos/7/isos/x86_64/CentOS-7-x86_64-DVD-1810.iso>`_。

hadoop 官方在 2.2.0 前默认只提供 32bit 安装包，其后只提供 64bit 安装包，这里选用 hadoop-2.7.5.tar.gz， 它可以运行在 64Bit CentOS 7 系统。

hadoop 使用 java 开发，但是为了提高性能底层代码使用 C 语言开发，这些库文件位于安装包的 lib/native 路径下：

.. code-block:: sh

  libhadoop.a  libhadooppipes.a  libhadoop.so  libhadoop.so.1.0.0  
  libhadooputils.a  libhdfs.a  libhdfs.so  libhdfs.so.0.0.0

可以使用 file 命令查看：

.. code-block:: sh
  
  [root@promote native]# file libhadoop.so.1.0.0 
  libhadoop.so.1.0.0: ELF 64-bit LSB shared object, x86-64, version 1 (SYSV), 
  dynamically linked, BuildID[sha1]=ed024ac48c0f542fa36ddc918a75c51e1c647424, not stripped

如果操作系统和软件 bit 位不匹配，则会在运行 hadoop 时报出如下错误信息：

.. code-block:: sh
  
  util.NativeCodeLoader: Unable to load native-hadoop library for your platform... 

如果要使用官方未提供的版本，需要配置 maven 环境，并使用源码编译，这个过程非常漫长。所以通常采用 64Bit 操作系统配合相应的 hadoop 官方版本。

环境配置
~~~~~~~~~~~~~

hadoop 支持 3 中配置模式，为了验证集群模式，这里使用虚拟机配置两台 CentOS 虚拟机。在实际的生产环境，通常使用 PXE 来批量安装操作系统，除了 IP 地址和主机名之外，所有操作系统配置应保持一致。以下配置均在 root 用户模式下进行。

环境配置如下：

- 主机名 hadoop0（192.168.10.7）用于 master。
- 主机名 hadoop0（192.168.10.8）用于 slave。

均添加普通用户 hadoop，并设置无密码 ssh 登录。以下为配置主机 hadoop0 为例。 

网络配置
````````````

网络配置包括静态 IP 地址，DNS，网关和主机名配置。首先明确主机需要配置的网络信息，通常这些信息会使用主机 MAC 地址生成，并打印成铭牌附在主机上以方便定位，例如：

.. code-block:: sh
  
  IP Address: 192.168.10.7
  Netmask: 255.255.255.0
  Gateway (Router): 192.168.10.1
  DNS Server 1: 192.168.10.1
  DNS Server 2: 8.8.8.8
  Domain Name: hadoop

集群服务器为了保证网络的稳定性，通常使用静态 IP，而不是动态 IP ，系统默认为动态 IP 地址。

.. code-block:: sh
  
  # ifconfig 
  ens33: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
          inet 192.168.10.8  netmask 255.255.255.0  broadcast 192.168.10.255
          inet6 fe80::ed0:8205:a345:6ea1  prefixlen 64  scopeid 0x20<link>
          ether 00:0c:29:d0:81:b0  txqueuelen 1000  (Ethernet)
          RX packets 189442  bytes 270275757 (257.7 MiB)
          RX errors 0  dropped 0  overruns 0  frame 0
          TX packets 33656  bytes 2325644 (2.2 MiB)
          TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0
  
ifconfig 查看网口名称，如果服务器配置有多块网卡，则注意连入集群中的网卡，或者做多网卡绑定操作。这里网卡对应网口名称为 ens33。

.. code-block:: sh
  
  # cd /etc/sysconfig/network-scripts
  # cp -f ifcfg-ens33 ifcfg-ens33.bak # 备份原配置文件是个好习惯

编辑 ifcfg-ens33 文件如下：

.. code-block:: sh
  
  # 指定网卡 MAC 地址
  HWADDR=00:0c:29:d0:81:b0 
  TYPE=Ethernet
  # 设置为静态 IP
  BOOTPROTO=staitc
  # 静态 IP 地址 
  IPADDR=192.168.10.7
  # 子网地址
  NETMASK=255.255.255.0
  # 网关
  GATEWAY=192.168.10.1
  # DNS 地址 
  DNS1=192.168.10.1
  DNS2=8.8.8.8
  # 启动时激活 
  ONBOOT=yes

重启网卡，使新配置生效：

  # systemctl restart network

测试网络连通性，可以 ping 网关，如果可以连接外网，可以 ping 外部网站，例如 www.baidu.com：

.. code-block:: sh
  
  # ping -c 1 192.168.10.1
  PING 192.168.10.1 (192.168.10.1) 56(84) bytes of data.
  64 bytes from 192.168.10.1: icmp_seq=1 ttl=64 time=2.05 ms

配置主机名：

.. code-block:: sh
  
  # 查看主机名
  # hostnamectl status
     Static hostname: localhost.localdomain
  Transient hostname: promote.cache-dns.local

  # 设置主机名
  # hostnamectl set-hostname hadoop0

以上配置修改 /etc/hostname 文件，如果直接修改该文件，则需要重启才能生效，测试主机名：

.. code-block:: sh
  
  # ping -c 1 hadoop0
  PING hadoop0 (192.168.10.7) 56(84) bytes of data.
  64 bytes from promote.cache-dns.local (192.168.10.8): icmp_seq=1 ttl=64 time=0.129 ms

关闭防火墙
```````````

由于 hadoop 会提供各类网络服务用于浏览存储和处理信息，主从节点之间也需要网络通信，这些均会创建动态端口。另外集群在和外部网络连接之间均需通过企业防火墙，所以为方便配置，需要关闭防火墙。

CentOS 7 默认使用 firewall 作为防火墙:

.. code-block:: sh
  
  # 查看防火墙状态
  # firewall-cmd --state
  running

  # 停止firewall
  # systemctl stop firewalld.service 
  
  # 重启防火墙使配置生效
  # systemctl restart iptables.service 

  # 禁止firewall开机启动
  # systemctl disable firewalld.service 
  #设置防火墙开机启动
  systemctl enable iptables.service 

CentOS 6 版本使用 iptables 设置防火墙，CentOS 7 也可以使用 yum -y install iptables-services 来安装 iptables 服务，

.. code-block:: sh
  
  # 查看防火墙状态
  # service iptables status
  
  # 关闭防火墙
  # service iptables stop
  # 开启防火墙
  # service iptables start
  
  # 重启防火墙
  # service iptables restart
  
  # 关闭防火墙开机启动
  # chkconfig iptables off
  # 开启防火墙开机启动
  # chkconfig iptables on

关闭 SELinux
````````````

SELinux 提供了程序级别的安全控制机制，hadoop 有些服务，例如 Ambari 需要关闭它：

.. code-block:: sh
  
  # 查看 SELinux 的状态 
  # getenforce
  Enforcing
  # 查看详细信息
  # sestatus  
  SELinux status:                 enabled
  SELinuxfs mount:                /sys/fs/selinux
  SELinux root directory:         /etc/selinux
  ......
  
  # 临时关闭
  # setenforce 0
  # 设置为 enforcing 模式
  # setenforce 1 

永久关闭需要修改配置文件 /etc/selinux/config，将其中SELINUX 设置为 disabled 并重启系统。

域名映射
``````````````

通过添加内网域名映射，可以直接使用域名互访主机。编辑 /etc/hosts，追加主机 IP 和主机名信息：

.. code-block:: sh
  
  192.168.10.7 hadoop0
  192.168.10.8 hadoop0

所有主机均复制相同的一份配置。

时间同步
``````````

在集群分布模式，由于主从节点基于时间来进行心跳同步，必须进行时间同步。在进行时间设置时必须调整时区，在安装操作系统时会设定它：

.. code-block:: sh
  
  # 查看时区状态
  # timedatectl status
  # 列出所有时区
  # timedatectl list-timezones 
  
  # 将硬件时钟调整为与本地时钟一致, 0 为设置为 UTC 时间
  # timedatectl set-local-rtc 1 
  
  # 设置系统时区为上海
  # timedatectl set-timezone Asia/Shanghai 

如果不考虑各个 CentOS 发行版的差异，可以直接这样操作：

.. code-block:: sh
  
  # cp /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

date 命令手动指定系统时间：

.. code-block:: sh
  
  # date -s "2018-05-13 12:01:30"

修改时间后，需要写入硬件 bios，这样在重启之后时间不会丢失：

.. code-block:: sh
  
  # hwclock -w

如果主机可以访问外网，推荐使用 ntp 服务同步系统时间，这样时间同步比较准确：

.. code-block:: sh
  
  # 命令格式 ntpdate ntp-server-ip
  # ntpdate ntp1.aliyun.com

当然也可以自行在内网搭建 ntp 服务器。

系统运行级别
````````````````

图形界面会耗费系统大量资源，为了提高性能，需要运行在非图形界面，也即多用户模式 3：

.. code-block:: sh

  # 查看当前运行级别
  # systemctl get-default
  
  # 设置默认运行级别，graphical.target 或者 multi-user.target
  # systemctl set-default TARGET.target
  
  # 设置为多用户级别
  # systemctl set-default multi-user.target

graphical.target 和 multi-user.target 分别对应 5 和 3，默认应该设置为多用户级别。

CentOS 7 默认使用 systemd 服务，可以通过 ps 查看进程，此时不再使用 /etc/inittab 文件来决定系统运行级别。

用户配置
``````````

基于安全考虑，大多数应用软件应该运行在普通用户状态，所以这里添加普通用户 hadoop，密码初始化为 123456：

.. code-block:: sh
  
  # useradd hadoop
  # passwd hadoop
  Changing password for user hadoop.
  New password: 
  BAD PASSWORD: The password is shorter than 8 characters
  Retype new password: 
  passwd: all authentication tokens updated successfully.

给与 hadoop 用户 sudoer 权限，可以让普通用户通过 sudo 修改系统文件或执行系统命令：
 
.. code-block:: sh
  
  # vi /etc/sudoer
  ## Allow root to run any commands anywhere
  root    ALL=(ALL)       ALL
  # 添加行
  hadoop  ALL=(ALL)       ALL

  # 切换用户以进行测试
  [root@promote ~]# su hadoop
  [hadoop@hadoop0 root]$ 

免密登录
```````````

由于 hadoop 的 shell 脚本均是通过 ssh 来统一在主从节点上执行的，所以必须配置免密码登录。

首先切换到普通用户，在所有主机上生成密钥，然后把生成的公钥分发给其他主机。

.. code-block:: sh
  
  # 通过 -t 和 -P 非交互模式生成密钥
  $ ssh-keygen -t rsa -P "" -f ~/.ssh/id_rsa
  Generating public/private rsa key pair.
  Created directory '/home/hadoop/.ssh'.
  Your identification has been saved in /home/hadoop/.ssh/id_rsa.
  Your public key has been saved in /home/hadoop/.ssh/id_rsa.pub.
  The key fingerprint is:
  SHA256:uCZ92HSkh3fvvFxp2+wS7dHIXRgS3uyQ+XEdt3tf7e0 hadoop@hadoop0
  The key's randomart image is:
  +---[RSA 2048]----+
  |            .. ..|
  |           ..=. =|
  |          . =.++o|
  |       . +   +.o+|
  |      . S + ..o=*|
  |     . = + . .+oX|
  |    . = o     .=*|
  |     o .     +o++|
  |              ==E|
  +----[SHA256]-----+

查看生成的密钥，其中 .pub 文件为公钥：

.. code-block:: sh

  $ ll ~/.ssh/
  total 8
  -rw------- 1 hadoop hadoop 1675 May 25 22:07 id_rsa
  -rw-r--r-- 1 hadoop hadoop  396 May 25 22:07 id_rsa.pub

所有当前主机可以免密登录的其他主机的公钥均放在 ~/.ssh/authorized_keys 文件中，本机登录自身也需要将公钥添加到 authorized_keys 文件中：

  $ cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys 
  
  # 测试本机登录
  [hadoop@hadoop0 .ssh]$ ssh hadoop0
  Last login: Sat May 25 21:14:25 2018 from hadoop0

所以可以分别复制所有 .pub 文件然后追加到某个主机的 authorized_keys 文件中，然后再分发 authorized_keys 文件。

ssh-copy-id 命令可以将本机的 .pub 追加到目标主机的 authorized_keys 文件中：

.. code-block:: sh

  $ ssh-copy-id hadoop0
  /usr/bin/ssh-copy-id: INFO: attempting to log in with the new key(s), to filter out any that are already installed
  /usr/bin/ssh-copy-id: INFO: 1 key(s) remain to be installed -- if you are prompted now it is to install the new keys
  hadoop@hadoop0's password: 
  
  Number of key(s) added: 1
  
  Now try logging into the machine, with:   "ssh 'hadoop0'"
  and check to make sure that only the key(s) you wanted were added.
  
  # 登录测试
  hadoop@hadoop0:/home$ ssh hadoop0
  Last login: Sat May 25 22:20:12 2019 from hadoop0
  [hadoop@hadoop0 ~]$ 

由于在分布式集群模式下，hadoop 命令可以在任一主机上执行并唤醒其他主机进程，所有主机生成的 .pub 文件必须分发给所有其他主机，这样主机之间才能任意互访。

软件安装
~~~~~~~~~

由于 hadoop 使用 java 编写，需要运行在 Java 虚拟机上，首先配置 JDK 环境。

安装 JDK
```````````

CentOS 默认安装 OpenJDK，首先需要把它卸载掉：

.. code-block:: sh
  
  [root@hadoop0 ~]# java -version
  openjdk version "1.8.0_212"
  OpenJDK Runtime Environment (build 1.8.0_212-b04)
  OpenJDK 64-Bit Server VM (build 25.212-b04, mixed mode)

查询 java 安装包，然后删除：

.. code-block:: sh
  
  # 以下四个文件需要删除
  [root@hadoop0 ~]# rpm -qa | grep openjdk
  java-1.7.0-openjdk-1.7.0.111-2.6.7.8.el7.x86_64
  java-1.8.0-openjdk-1.8.0.102-4.b14.el7.x86_64
  java-1.8.0-openjdk-headless-1.8.0.102-4.b14.el7.x86_64
  java-1.7.0-openjdk-headless-1.7.0.111-2.6.7.8.el7.x86_64
  
  # 使用 rpm -e --nodeps 依次删除
  [root@hadoop0 ~]# rpm -e --nodeps java-1.7.0-openjdk-1.7.0.111-2.6.7.8.el7.x86_64
  ......
  
  # 验证删除完毕
  [root@hadoop0 ~]# jave -version
  bash: jave: command not found...

这里使用 1.8 版本的 Oracle 官方 64Bit JDK jdk-8u172-linux-x64.tar.gz。

.. code-block:: sh
  
  [root@hadoop0 hadoop]# mkdir /lib/jdk/
  [root@hadoop0 hadoop]# tar zxf jdk-8u172-linux-x64.tar.gz -C /opt/
  
在 /etc/profile 在中添加系统环境变量，使得所有用户均可使用；如果限定某个用户使用，则添加环境变量到对应用户的 ~/.bash_profile 文件中。 

.. code-block:: sh
     
  export JAVA_HOME=/opt/jdk1.7.0_80
  export PATH=$PATH:$JAVA_HOME/bin
  
  # souce 执行脚本使其立即生效
  # source /etc/profile
  
  # 验证 JDK 是否安装成功
  # java -version
  java version "1.8.0_172"
  Java(TM) SE Runtime Environment (build 1.8.0_172-b11)
  Java HotSpot(TM) 64-Bit Server VM (build 25.172-b11, mixed mode)

安装 hadoop
```````````````

由于 hadoop 以普通用户权限运行，所以安装时也使用普通用户，首先切换到普通用户 su hadoop。为了方便修改 hadoop 的配置文件，解压到 hadoop 用户的 home 目录下，这样可以避免使用超级用户权限修改配置文件。

.. code-block:: sh

  [hadoop@hadoop0 ~]$ sudo tar zxf  hadoop-2.7.5.tar.gz -C ~/
  [sudo] password for hadoop
  
为 hadoop 添加环境变量，编辑 /etc/profile 文件：
  
  [hadoop@hadoop0 ~]$ sudo vi /etc/profile
  export HADOOP_HOME=/home/hadoop/hadoop-2.7.5
  export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin

由于 hadoop 进程均是后台启动，所以 shell 中的 JAVA_HOME 环境变量无法被读取，必须通过 etc/hadoop/hadoop-env.sh 设置：

.. code-block:: sh

  # 设置和 /etc/profile 中保持一致：
  export JAVA_HOME=/opt/jdk1.8.0_172

souce 命令必须在 root 用户下执行：

.. code-block:: sh

  [hadoop@hadoop0 ~]$ sudo su
  [root@hadoop0 hadoop]# source /etc/profile
  [root@hadoop0 hadoop]# exit

  # 验证安装环境
  [hadoop@hadoop0 ~]$ hadoop version
  Hadoop 2.7.5

运行模式
~~~~~~~~~~~

Hadoop 有三种运行模式：单机模式（Standalone Mode），伪分布模式（Pseudo-Distrubuted Mode）和全分布式集群模式（Full-Distributed Mode）。

单机模式是 Hadoop 安装完后的默认模式，无需进行任何配置。另外针对 hadoop 的所有配置均位于 etc/hadoop 中的 xml 文件中。

单机模式
```````````

单机模式也被称为独立模式，主要用于开发和调式，不对配置文件进行修改，不会使用 HDFS 分布式文件系统，而直接使用本地文件系统。

同样，hadoop 也不会启动 namenode、datanode 等守护进程，Map 和 Reduce 任务被作为同一个进程的不同部分来执行的，以验证 MapReduce 程序逻辑，确保正确。

官网提供了单词统计操作示例，用于验证单机模式，注意 output 文件不可以存在，否则输出报错。

.. code-block:: sh

  [hadoop@hadoop0 ~]$ mkdir input
  [hadoop@hadoop0 ~]$ cd input/
  [hadoop@hadoop0 input]$ echo "hello world" > test.txt
  [hadoop@hadoop0 input]$ cd ../
  [hadoop@hadoop0 ~]$ hadoop jar hadoop-2.7.5/share/hadoop/mapreduce/hadoop-mapreduce-examples-2.7.5.jar wordcount input output
  
这里创建只包含 "hello world" 两个单词的测试文件 test.txt，以便验证结果正确性，查看 output 文件：

.. code-block:: sh
  
  [hadoop@hadoop1 ~]$ cd output/
  [hadoop@hadoop1 output]$ ll
  总用量 0
  -rw-r--r-- 1 hadoop hadoop 16 5月  26 11:54 part-r-00000
  -rw-r--r-- 1 hadoop hadoop 0 5月  26 11:54 _SUCCESS

_SUCCESS 文件用于指示任务运行成功，是一个标记文件，没有内容。part-r-0000 存储结果：

.. code-block:: sh

  [hadoop@hadoop1 output]$ cat part-r-00000 
  hello   1
  world   1

单机模式使用本地文件系统，可以使用 hadoop fs 命令查看：

.. code-block:: sh
  
  # 查看文件系统
  [hadoop@hadoop1 ~]$ hadoop fs -df
  Filesystem        Size        Used   Available  Use%
  file:///    8575254528  6253735936  2321518592   73%
  
  # 当前文件夹文件列表
  [hadoop@hadoop1 ~]$ hadoop fs -ls
  Found 16 items
  -rw-------   1 hadoop hadoop       2600 2019-05-26 11:39 .bash_history
  -rw-r--r--   1 hadoop hadoop         18 2018-10-31 01:07 .bash_logout
  ......

伪分布模式
``````````````

伪分布式只需要一台主机，这里使用 hadoop1 主机为例。

核心配置文件 etc/hadoop/core-site.xml 配置主节点 namenode:

.. code-block:: sh

  <configuration>
      <property>
          <name>fs.defaultFS</name>
          <value>hdfs://hadoop1:9000</value>
      </property>
      <property>
          <name>hadoop.tmp.dir</name>
          <value>/home/hadoop/hadoop-2.7.5/tmp</value>
      </property>
  </configuration>

- fs.defaultFS 属性指定 namenode 的 hdfs 协议的文件系统通信地址，格式为：协议://主机:端口。
- hadoop.tmp.dir 指定 hadoop 运行时的临时文件存放目录（tmp 文件夹已使用 mkdir 创建）。

hdfs-site.xml 配置分布式文件系统的相关属性：

.. code-block:: sh
  <configuration>
      <property>
          <name>dfs.namenode.name.dir</name>
          <value>/home/hadoop/data/name</value>
      </property>
      <property>
          <name>dfs.datanode.data.dir</name>
          <value>/home/hadoop/data/data</value>
      </property>
      <property>
          <name>dfs.replication</name>
          <value>1</value>
      </property>
  </configuration>

- dfs.namenode.name.dir 和 dfs.datanode.data.dir 分别配置主从节点的存储位置，默认位置为 /tmp/hadoop-${usrname}/dfs/。/tmp 是临时文件夹，空间可能会被系统回收。
- dfs.replication 属性指定每个 block 的冗余副本个数，在伪分布模式下配置为 1 即可，也即不启用副本。

yarn-site.xml 用于配置资源管理系统 yarn ：

.. code-block:: sh

  <configuration>
      <property>
          <name>yarn.resourcemanager.hostname</name>
          <value>hadoop1</value>
      </property>
      <property>
          <name>yarn.nodemanager.aux-services</name>
          <value>mapreduce_shuffle</value>
      </property>
  </configuration>

- yarn.resourcemanager.hostname 配置主资源管理器 resourcemanager 的主机名。
- yarn.nodemanager.aux-services 指明提供 mapreduce 服务。

mapred-site.xml 指定 mapreduce 运行的资源调度平台为 yarn：

.. code-block:: sh
  
  # 从模板文件复制，然后编辑
  $ cp -f mapred-site.xml.template mapred-site.xml
  
  <configuration>
      <property>
      <name>mapreduce.framework.name</name>
      <value>yarn</value>
      </property>
  </configuration>

配置 salves，指定 datanode 主机名。

.. code-block:: sh
  
  hadoop1

格式化 hdfs：

.. code-block:: sh
  
  # 原命令 hadoop namenode -formate 被更新为
  $ hdfs namenode -format
  
查看格式化后的 HDFS 文件系统，位于 /home/hadoop/data/name 下：

.. code-block:: sh

  [hadoop@hadoop1 data]$ tree
  .
  └── name # 对应 NameNode 进程，存储主节点信息
      └── current
          ├── fsimage_0000000000000000000
          ├── fsimage_0000000000000000000.md5
          ├── seen_txid
          └── VERSION
  
  2 directories, 4 files

fsimage 文件是 namenode 中关于元数据的镜像，也称为检查点。

最后启动伪分布式集群的进程。

.. code-block:: sh

  $ start-dfs.sh
  
  # 查看启动进程
  $ jps
  13520 Jps
  12787 NameNode # 主节点进程
  13396 SecondaryNameNode # 助理进程
  12885 DataNode # 从节点进程 
  
  $ start-yarn.sh
  $ jps
  13712 Jps
  13681 NodeManager     # 从管理进程
  12787 NameNode
  13396 SecondaryNameNode
  12885 DataNode
  13581 ResourceManager # 主管理进程

也可以通过 WEB 页面查看进程是否启动成功：

- hdfs 服务地址 http://192.168.10.8:50070/
- yarn 服务地址 http://192.168.10.8:8088/

相应的退出进程脚本为：

.. code-block:: sh

  $ stop-dts.sh
  $ stop-yarn.sh

伪分布验证
```````````

这里依然使用字符统计示例，在 HDFS 文件系统中创建  wordcount/input 文件夹，然后存入 test.txt 文件。

.. code-block:: sh

  $ hadoop fs -mkdir -p /wordcount/input
  $ hadoop fs -ls -R /
  drwxr-xr-x   - hadoop supergroup   0 2019-05-26 17:23 /wordcount
  drwxr-xr-x   - hadoop supergroup   0 2019-05-26 17:23 /wordcount/input

使用 put 命令追加文件：

.. code-block:: sh

  $ hadoop fs -put test.txt /wordcount/input/
  $ hadoop fs -ls /wordcount/input/
  Found 1 items
  -rw-r--r--   1 hadoop supergroup   12 2019-05-26 17:30 /wordcount/input/test.txt

  # 查看 HDFS 目录
  [hadoop@hadoop1 data]$ tree
  .
  ├── data  # 对应 DataNode 进程，存储 block 数据
  │   └── current
  │       └── BP-1621093575-192.168.10.8-1558860568281
  │           ├── current
  │           │   ├── dfsUsed
  │           │   ├── finalized
  │           │   └── rbw
  │           └── tmp
  └── name
      └── current
          ├── fsimage_0000000000000000000
          ├── fsimage_0000000000000000000.md5
          ├── seen_txid
          └── VERSION  

统计单词：

.. code-block:: sh
  
  $ hadoop jar hadoop-2.7.5/share/hadoop/mapreduce/hadoop-mapreduce-examples-2.7.5.jar \
  wordcount /wordcount/input/ /wordcount/output

  # 查看输出结果
  $ hadoop fs -ls -R /wordcount/output
  -rw-r--r--   1 hadoop supergroup          0 2019-05-26 17:40 /wordcount/output/_SUCCESS
  -rw-r--r--   1 hadoop supergroup         16 2019-05-26 17:40 /wordcount/output/part-r-00000

  $ hadoop fs -cat /wordcount/output/part-r-00000
  hello   1
  world   1

使用 get 下载文件：

.. code-block:: sh
  
  $ hadoop fs -get /wordcount/output/* output/ 

