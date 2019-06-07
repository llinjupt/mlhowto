scala
================

Scala 是一种结合函数式编程和面向对象的纯面向对象语言，所以被称为多范式语言。它和 java 语言类似，需要进行编译并运行在 JVM 虚拟机上。

Scala 语言具有如下特点：

- 优雅：这是框架设计师第一个要考虑的问题，框架的用户是应用开发程序员，API是否优雅直接影响用户体验。
- 开发速度快：Scala 语言表达能力强，相对于 Java 开发速度快
- 运行速度快：Scala 是静态编译的，所以和 Python，Ruby 等解释性语言比，速度快很多。

随着 Spark 成为 Hadoop 生态圈的数据处理主力，而 Spark 是使用 Scala 语言开发的，如果要深入理解和定制 Spark，就要熟悉 Scala。

Scala 的源文件被命名为 .scala，通过编译器 scalac 编译为 .class 字节码文件。

环境安装
-----------

安装 scala
~~~~~~~~~~~~

因为 Scala 是运行在JVM平台上的，所以安装 Scala 之前要安装 JDK，注意安装时路径不要有空格或者中文。

访问`Scala官网 <http://www.scala-lang.org>`_ 下载 Scala 编译器安装包，由于目前大多数框架都是用 2.10.x 编写开发的，推荐安装 2.10.x 版本，Windows 平台直接下载 scala-2.10.6.msi 安装即可，会自动配置环境变量。

验证安装环境：

.. code-block:: sh

  E:\>scala -version
  Scala code runner version 2.10.6 -- Copyright 2002-2018, LAMP/EPFL and Lightbend, Inc.

Linux 环境下载 .tgz 文件，解压后在 /etc/profile 下修改环境变量

.. code-block:: sh

  # 解压缩
  $ tar -zxvf scala-2.10.6.tgz -C /opt/

  vi /etc/profile
  export JAVA_HOME=/opt/jdk1.7.0_80
  export PATH=$PATH:$JAVA_HOME/bin:/opt/scala-2.10.6/bin

安装 Idea IDE
~~~~~~~~~~~~~

Idea 是用户开发 Java 项目的优秀IDE，安装 Scala 插件后可以支持 Scala 的高效开发。

从 http://www.jetbrains.com/idea/download/ 下载社区免费版并安装，安装时如果有网络可以选择在线安装 Scala 插件。

如果网速较慢，可以选择离线安装，从地址 http://plugins.jetbrains.com/?idea_ce 搜索 Scala 插件，然后下载。

.. figure:: imgs/scala/idea.png
  :scale: 80%
  :align: center
  :alt: idea

  首次启动窗口

首次启动 Idea 安装Scala插件：Configure -> Plugins -> Install plugin from disk -> 选择Scala插件 -> OK -> 重启IDEA。

如果当前已经进入 Idea，可以通过 File->Settings 搜索 Plugins 标签页，在标签页面右下角选择 Install plugin from disk，然后从本地磁盘安装插件。

sbt 配置
~~~~~~~~~

sbt 用于 scala 项目的自动编译，类似 java 项目中的 maven。从 `sbt官网 <ttp://www.scala-sbt.org/download.html>`_ 下载安装包，最新版本为 sbt-1.2.8.tgz。

.. code-block:: sh

  $ tar zxvf sbt-1.2.8.tgz
  
  # 解压后目录为 /home/hadoop/sbt
  # root 权限在 /etc/profile 中配置环境变量
  export SBT_HOME=/home/hadoop/sbt
  export PATH=$PATH:${SBT_HOME}/bin 
  
  # 使能环境变量
  $ source /etc/profile
  
  # 第一次执行将会下载相关依赖，存储在用户目录 ~/.ivy2 下
  $ sbt   
  ......
  sbt:hello> about
  [info] This is sbt 1.2.8
  sbt:hello> help # help 查看帮助信息，支持 tab 补全
  sbt:hello> exit # 退出交互环境

仓库路径可以在 conf/sbtopts 配置文件中修改。~/.sbt 目录下放置 sbt 的全局配置，和 scala 语言包。Window 环境需要修改 sbtconfig.txt，查看 bin/sbt.bat 可以看到它没有使用 sbtopts 文件。

为了加速依赖包的下载，可以在 ~/.sbt/repositories 配置阿里云作为源：

.. code-block:: sh

  [repositories]
    local
    nexus-aliyun:http://maven.aliyun.com/nexus/content/groups/public

创建名为 hello 的文件夹，并生成 hello.scala 源码以及 build.sbt：

.. code-block:: sh

  hadoop@hadoop0:~/hello$ tree
  .
  ├── build.sbt
  └── hello.scala
  
build.sbt 是编译配置文件，类似 gcc 中的 Makefile，hello.scala 内容为：

.. code-block:: scala
  :linenos:
  :lineno-start: 0

  object Hello {
      def main(args: Array[String]) = println("Hello world!")
  }

然后在目录下运行 sbt 命令，进入交互式命令行环境：
  
.. code-block:: sh
  
  $ sbt
  sbt:hello> compile # 编译，支持增量编译
  [info] Updating ...
  [info] Done updating.
  [info] Compiling 1 Scala source to ...
  [info] Done compiling.
  [success] Total time: 3 s, completed Jun 7, 2018 1:19:56 PM
  
  sbt:hello> run     # 执行
  [info] Packaging ...
  [info] Done packaging.
  [info] Running Hello # 入口对象
  Hello world!         # 执行结果 
  [success] Total time: 1 s, completed Jun 7, 2018 1:20:51 PM

run 操作依赖于compile，如果没有编译，则会先执行 compile，然后执行 main 方法。 sbt 支持在源码变动时自动执行命令，只需在命令前添加 ~ 符号。

我们并没有编辑 build.sbt，sbt 完全按照约定工作。sbt 将会自动找到以下内容：

- 项目根目录下的源文件
- src/main/scala 或 src/main/java 中的源文件
- src/test/scala 或 src/test/java 中的测试文件
- src/main/resources 或 src/test/resources 中的数据文件
- lib 中的 jar 文件 

默认情况下，sbt 会用和启动自身相同版本的 Scala 来构建项目。通常采用以下方式安排 sbt 项目的目录结构：

.. code-block:: sh

  build.sbt
  src/
      main/
          resources/        # 数据文件
          scala/            # scala 源文件
          java/             # java 源文件
      test/
          resources/        # 数据文件
          scala/            # scala 测试源文件
          java/             # java 测试源文件
  lib                       # jar文件
  project
      build.properties
      plugins.sbt 

然后在 src/main/scala 下放置所有包的源文件。这里在 src/main/scala/example 下创建 hello.scala 文件：

.. code-block:: scala
  :linenos:
  :lineno-start: 0

  package example
  
  object Hello {
      def main(args: Array[String]) = println("Hello world!")
  }

编辑 build.sbt，创建构建配置：

.. code-block:: sh

  lazy val root = (project in file("."))
    .settings(
      name := "hello",  # 项目工程名
      version := "1.0", # 项目版本
      scalaVersion := "2.12.7" # 依赖的 scala 版本
    )

以通过创建 hello/project/build.properties 文件强制指定一个版本的 sbt。默认使用当前版本。 可以使用 run 加入口类运行，也可以通过 console 命令进入 scala 运行：

.. code-block:: sh

  sbt:hello> run example.Hello
  [info] Running example.Hello example.Hello
  Hello world!
  [success] Total time: 1 s, completed Jun 7, 2018 4:26:20 PM
  
  # 进入 scala 交互环境
  sbt:hello> console
  [info] Starting scala interpreter...
  Welcome to Scala 2.12.7 (Java HotSpot(TM) Server VM, Java 1.8.0_31).
  Type in expressions for evaluation. Or try :help.
  
  scala> import example.Hello
  import example.Hello
  
  scala> Hello.main(null)
  Hello world!

package 用于打包 jar，可以使用 scala 解析器直接执行该 jar 包文件：

.. code-block:: sh

  sbt:hello> package
  [info] Updating ...
  [info] Done updating.
  [info] Compiling 1 Scala source to /home/hadoop/sbtproject/hello/target/scala-2.12/classes ...
  [info] Done compiling.
  [info] Packaging /home/hadoop/sbtproject/hello/target/scala-2.12/hello_2.12-1.0.jar ...
  [info] Done packaging.
  [success] Total time: 1 s, completed Jun 7, 2018 4:30:31 PM

  # 直接执行 jar
  $ scala target/scala-2.12/hello_2.12-1.0.jar
  Hello world!

创建 spark 项目
~~~~~~~~~~~~~~~~~

以 WordCount 单词统计为示例，过程如下：

.. code-block:: sh

  # 创建 WordCount
  $ mkdir WordCount
  $ cd WordCount
  $ mkdir -p src/main/scala/example
  
在路径 src/main/scala/example 下创建 WordCount.scala，内容如下：

.. code-block:: scala
  :linenos:
  :lineno-start: 0
  
  package example
  import org.apache.spark.SparkContext
  
  object WordCount {
      def main(args: Array[String]): Unit = {
        val inputPath = args(0)     // 输入文件夹
        val outputPath = args(1)    // 输出文件夹
        val sc = new SparkContext()
        val lines = sc.textFile(inputPath)
        val wordCounts = lines.flatMap {line => line.split(" ")}
                         .map(word => (word, 1)).reduceByKey(_ + _)
        wordCounts.saveAsTextFile(outputPath)
    }
  }

最后配置 build.sbt 构建文件，指明依赖 spark-core：

.. code-block:: scala
  :linenos:
  :lineno-start: 0
  
  lazy val root = (project in file(".")).
    settings(
    name := "WordCount",
    version := "1.0",
    scalaVersion := "2.11.12", // 这里要和 spark 中使用的 scala 版本一致
    libraryDependencies += "org.apache.spark" %% "spark-core" % "2.4.3" % "provided"
  )
  
注意，如果 scala 使用版本不一致，运行 spark-submit 提交任务时将出现 java.lang.NoClassDefFoundError。以本地模式运行测试：

.. code-block:: sh

  # 创建测试路径和文件
  $ tree /home/hadoop/input/
  /home/hadoop/input/
  └── test.txt 

  $ cat /home/hadoop/input/test.txt 
  hello world
  
  # 测试单词统计数据包
  $ spark-submit --master local[4] --class example.WordCount --executor-memory 512m \
    target/scala-2.11/wordcount_2.11-1.0.jar \
   /home/hadoop/input/ /home/hadoop/out

  # 查看统计结果
  $ cat /home/hadoop/out/part-00000
  (hello,1)
  (world,1)
  
基本语法
----------

HelloWorld
~~~~~~~~~~~

使用原生方式直接编写，并编译，然后执行。

.. code-block:: scala
  :linenos:
  :lineno-start: 0
  
  # 创建 HelloWorld.scala
  object HelloWorld {
  	  def main(args: Array[String]):Uint={
  		    println("Hello World!");
  		}
  }
  
  # scalac HelloWorld.scala 编译生成 HelloWorld.class 文件
  # scala HelloWorld  执行

由于 scalac 没有进行 java 库的链接，不能直接使用 java HelloWorld 执行。

