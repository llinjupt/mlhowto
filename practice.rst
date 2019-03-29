机器学习实战
=============

本部分实战用例主要是对 Adrian Rosebrock博客 `pyimagesearch <https://www.pyimagesearch.com>`_，OpenCV 官网，19Channel 等提供实例的总结和验证，主要集中在计算机视觉领域。

实战主要聚焦在如下几个部分：
- 模型的应用（目标检测(Object Detection)，多目标检测，实时检测）
- 模型的训练（数据收集，提取，归一化，训练，各类网络的识别）
- 模型性能对比和算法改进（比较耗时，占比较少）
- 嵌入式应用（树莓派/BeagleBoard，手机应用）

环境安装
-----------

caffe
~~~~~~~~~~

尽管 tensorflow 和 pytorch 渐渐成为深度学习框架的主流，如果你拿到一个模型是基于其他框架训练而来的，如果要进行验证就需要相应的环境。好在跨平台的 Anaconda 提供了这一方便（令人稍许轻松）。

和其他计算机应用领域类似，配置环境这种体力密集型劳动在人工智能领域也不能幸免（AI 就是 AI，不是真正的I ^>^），甚至更甚（由于AI的快速发展，硬件算力不停升级，驱动不停更新，各类算法也层出不穷，所以软件框架也就不停更新，同时类似 Python 的胶水语言也在不停变动，导致版本依赖很强）。

Caffe（Convolutional Architecture for Fast Feature Embedding）是一种常用的深度学习框架，主要应用在视频、图像处理方面的应用上。

这里以 WIN10 上的 Anaconda 为例（强烈建议使用 Linux 操作系统，特别是 Ubuntu，大部分开源社区的成员对 Opensource 系统怀有异常的热情，你将能得到更好的帮助），在 Python 环境中配置 caffe。

1. 使用 conda 创建 caffe 的 Python 应用环境，由于 caffe 指定依赖 Python 2.7 或者 Python 3.5，所以要为它另起炉灶（这也是为什么推荐 Anaconda 的原因：支持不同 Python 版本环境，且提供了各类机器学习库的源）。cmd 窗口查看当前 conda 的环境：

.. code-block:: sh
  :linenos:
  :lineno-start: 0
  
  > conda env list
  # conda environments:
  #
  base                     E:\Anaconda3

笔者环境存在 base 环境，支持较新的 Python3.6。所以不满足 caffe 对 Python 版本的需求。创建 caffe-py3.5 环境：

.. code-block:: sh
  :linenos:
  :lineno-start: 0
  
  > conda create -n caffe-py3.5 python=3.5
  > conda env list
  # conda environments:
  #
  base                     E:\Anaconda3
  # 新建的caffe-py3.5 环境，路径放置在 envs 目录下
  caffe-py3.5           *  E:\Anaconda3\envs\caffe-py3.5 

在环境创建过程中，会安装一些最基本的程序包。成功后切换到新建的环境 Python3.5 环境：

.. code-block:: sh
  :linenos:
  :lineno-start: 0
  
  > activate caffe-py3.5
  > python --version
  
  Python 3.5.4 :: Continuum Analytics, Inc.

2. 安装 caffe 依赖，必须要注意 protobuf==3.1.0 版本：

.. code-block:: sh
  :linenos:
  :lineno-start: 0
  
  > conda install --yes cmake ninja numpy scipy protobuf==3.1.0 six scikit-image pyyaml pydotplus graphviz

3. 安装Windows 版 git 以下载 caffe 源码，注意源码放置为位置不要过深，也不要包含特殊字符，比如空格或者 . 之类字符，为了避免陷入奇怪编译的问题，建议放置在系统盘根目录下：

.. code-block:: sh
  :linenos:
  :lineno-start: 0
  
  > d:
  > git clone https://github.com/BVLC/caffe.git
  > git branch -a
    * master
    remotes/origin/HEAD -> origin/master
    remotes/origin/gh-pages
    remotes/origin/intel
    remotes/origin/master
    remotes/origin/opencl
    remotes/origin/readme_list_branches
    remotes/origin/tutorial
    remotes/origin/windows    
  > git checkout windows   # 切换到 windows 分支
  
切换到 windows 分支非常重要，否则根本无法编译。

4. 打开 VS2015 x86 x64 兼容工具命令提示符，并使用 conda 切换到caffe-py3.5环境。进入 caffe 目录，执行 cmake .，配置编译环境。

.. code-block:: sh
  :linenos:
  :lineno-start: 0
  
  > cd caffe
  > cmake .

cmake 默认使用 Ninja 编译器（速度比较快），但是可能出现找不到头文件的问题。笔者就遭遇了这种陷阱。

5. 编译，进入 caffe 下的 scripts 目录，执行 build_win.cmd 。如果使用默认的 Ninja 编译器遭遇 ninja: build stopped: subcommand failed. 

.. code-block:: sh
  :linenos:
  :lineno-start: 0
  
  编辑 build_win.cmd 将所有
  if NOT DEFINED WITH_NINJA set WITH_NINJA=1
  
  替换为
  if NOT DEFINED WITH_NINJA set WITH_NINJA=0

然后删除掉 scripts 目录下的 build 和 caffe 下的 CMakeFiles 和 CMakeCache.txt 文件，重新执行第 4 步。

6. 编译完毕后，执行 caffe 依赖的其他安装包，requirements.txt 位于 caffe\python 目录：

.. code-block:: sh
  :linenos:
  :lineno-start: 0
  
  > pip install -r requirements.txt

安装出现 leveldb 无法编译，可以在 requirements.txt 删除它，该库用于读取 Matlab 数据库文件，如果确实需要则需要手动编译安装。

7. 安装 caffe 到 Anaconda 环境。 复制 python\caffe 文件夹到 E:\Anaconda3\envs\caffe-py3.5\Lib\site-packages。书写 test.py 引用 caffe 进行测试。

不建议使用老版本或者不稳定版本的数据包，除非迫不得已。requirements 中需要 >= 版本都应该取等于，否则会出现依赖循环问题。

conda
~~~~~~~~~

conda 用于管理 Anaconda3 科学计算环境软件包。

环境管理
```````````````

.. code-block:: sh
  :linenos:
  :lineno-start: 0
  
  # 环境相关
  # 下面是创建python=3.6版本的环境，取名叫py36
  conda create -n py36 python=3.6

  # 删除环境
  conda remove -n py36 --all

  # 激活 py36 环境，windows 无需 source 命令前缀
  activate py36

  # 退出当前环境
  deactivate
  
  # 复制（克隆）已有环境
  conda create -n py361 --clone py36

  # 查看当前所有环境
  conda env list
  
创建的环境路径位于 Anaconda 安装文件的 envs 文件夹下。

软件包管理
``````````````

.. code-block:: sh
  :linenos:
  :lineno-start: 0
  
  # 查看当前环境信息
  conda info

  # 查看安装软件列表
  conda list
  
  # 查看软件包信息，软件包名称支持模糊查询
  conda list python

  # 查找软件包通道 channel 
  anaconda search -t conda pyqt5
  
  # 安装软件包到 py36 环境，如果不指定环境，则作用到当前环境
  conda install --name py36 numpy -c 指定通道
  
  # 删除软件包，如果不指定环境，则作用到当前环境 
  conda remove --name py36 numpy
  
  # 查询 conda 版本号
  conda --version

在启动 Anaconda Navigator 或者 Sypder 遇如下问题时：

.. code-block:: sh
  :linenos:
  :lineno-start: 0
  
  # ModuleNotFoundError: No module named 'PyQt5.QtWebKitWidgets'
  conda update -c conda-forge qt pyqt

写在前面
----------

相关软硬平台
~~~~~~~~~~~~~~

Intel OpenVINO /RealSense / Movidius
ARM   Tengine

NumPy 可以配置为使用线程数字处理器库（如MKL）。

移动端迁移学习方案
Apple turicreate CoreML ->iOS
Google Tensorflow -> Android

加速：cython or OpenMP https://www.openmp.org/

关于"AI应用"的歪思考
~~~~~~~~~~~~~~~~~~~~

使用模型训练（深度学习神经网络）的流程：采集数据，尽可能多的采集广泛的数据（采集范围根据需求确定，根据需要进行精确处理：数据清洗），并准确标注。训练，可以多模型调参，并对比性能，导出模型。在实际应用环境，采集到的数据必须进行同样的精确预处理，通过模型进行识别，大体流程：

- 数据采集，通常由程序自动完成，比如从大量不同类型的视频中采集人脸，然后通过人工剔除错误信息（否则再多数据都白给），关键点标注（关键点也可以由程序完成，但需要人工进行后期的精确调整）
- 数据处理，采集到的样本可能大小，颜色，所占图片位置不同，所以要进行精确处理。
- 选择合适的模型，或者多个模型以进行效果对比 
- 实际应用场景进行验证，性能，效果，然后把错误数据继续反馈到模型继续训练，提高模型的鲁棒性。

性能不达标：

- 错误率高 1.软调节：数据是否准确，规模是否足够大到能满足需求，训练数据够好，则更新算法 2.硬调节，更换更高更好的传感器，提高分辨率和响应速度
- 速度慢  1.软调节：升级模型算法（需要有所突破）或者根据具体场景，来缩小图片尺寸，代价是距离远了，识别率变差；或者并行改串行，多线程处理；硬调节，增加多传感器，对应多线程处理；升级CPU，升级GPU，升级DSP，升级FPGA，根据SOC厂家解决方案来定（工程量不小，开始原型预研就要估计好数据量，莫盲目乐观）。 

这看起来很有趣，但是有什么实际用处呢？这是一个好问题，一个关键问题！ 但是 Data talks！

我所居住的小区后面就是地铁口，巧合的是在北阳台透过窗户，就可以完全看到它，于是我就把一个摄像头对准了这个出入口，并统计从早上 6:00 到晚上 6:00 出入该地铁口的人流，尽管有些距离，通过调焦还是可以看清进出的每个人，这对于识别人群的个体很有帮助。通过收集的数据，可以轻松的获取这入口人流数据，可以想象如果可以统计多个地段出入口数据就可以大体估计出这个城市的通勤情况。如果有长期的数据统计，那么可以得到很多更有趣的统计信息，比如人流的潮汐现象，每天或者每个月不同时期进出人流情况。顺便可以分析下男女占比，甚至着装颜色，只要发挥想象力，甚至可以统计下多少人是从地铁口的早餐摊买食物，进而分析下这个摊点的盈利状况。

周末带着四螺旋桨遥控飞机陪着小朋友玩，无意中发现很多楼房的顶层都装有太阳能热水器，不妨统计下热水器的品牌分布。由于这一片都是新小区，所以这个分布能在一定程度上反应该品牌在该城市的受欢迎程度。如果能够对城市的不同区域的小区进行采样，这个数据的分布就要正确得多。

晚上带着小朋友在车库玩滑板车，通过遥控飞机在车库来回飞行，进行车辆品牌的识别，甚至车辆的型号，非常容易统计出各个品牌在该片区的销售情况，如果能把数据扩大到多个小区，那么这个分布就非常可信了。

突然湖边有一群野鸟从树丛中飞起，并向着对岸飞去，掏出手机拍照上传到我的微信小程序，它的后端就是云服务器，服务器上的识别程序告诉我一共有18只，虽然无法识别这是什么鸟类，却告诉了这一群飞鸟的数目，这在生态学研究中很重要（人工去统计种群数目成本昂贵）。如果要对一片野生动物栖息地里的动物进行数量统计，特别是草原地区，那么使用遥控飞机拍照识别是没有再简单省事的了。

远处是串流不息的大运河，并且过往船只繁多，在高楼上也可以看到，把数据采样分析，就可以知道这条水运路线的繁忙程度以及船只吨位的分布了，如果视频数据够清晰，还可以识别所载货物种类。我现在才知道很多加油站的燃油均是通过水路运输的。长期的数据积累将会反应出更有趣的真相，如果可以分析每条船的所属地区，那么就可以大概知道货物去往了哪里......

如果把这种应用放在人造卫星上，用途就更是大得多了（可以想见人造卫星上的大数据所能揭露的真相有多么惊人！）。当然在微观领域，也有很大的用途，比如识别和统计显微镜下的细胞或者细菌数量。

简单的颜色，形状甚至运动物体的识别无需人工智能的加持也可以工作得很好，但是复杂的事物识别就需要在大型机上训练好分类算法模型，比如手势，脸部识别，甚至表情识别，动态物体跟踪等等。更复杂环境下的识别就需要愈加复杂的模型和算力支持，并且要考虑实时性和耗能，比如智能驾驶和机器人领域。

当前阶段的人工智能远飞人们想象的智能，并且还相当遥远。大多数据的模型算法都是通过大量数据分析出其中的规律，所以只能算是统计模型。并且严重依赖严谨的准确的数据，而数学模型简单还是复杂对预测准确性并没有直接关系，只要模型正确，结果一定相差不大，都能正确反映出训练数据的模式规律。

无论简单还是复杂的人工智能算法都无法从不准确的大数据中分析出准确的规律，也不可能从准确大数据中分析出离谱的预测模型，否则这种模型早就被淘汰了。一定要相信能够在学术和应用领域流传至今的知名算法都是经过长期验证的。同时不要盲信那些准确率高达吓人地步的模型，没有透明的训练数据，测试数据，训练耗时以及算法的可控性，复杂度的同时对比，只有一个准确率有什么意义。

事实证明，不用的算法模型在准确性上除了与一些模型参数有关外，在相同的训练数据基础上，结果都是大同小异。很多准确率宣称 99% 的模型一旦拿到实际的应用环境，其结果就连作者自己都大跌眼镜。为什么会出现这种情况？它与训练数据的真实的有效值（ground truth）到底是多少有关。一个数据集常常使用相同的方式（局限于特定的采集软件或者人工来采样生成）来获取，一部分用来训练，一部分用来验证，其结果只在这非常局限的缺乏真实应用环境的有效值上表现很好，有什么用呢？

可以看到无数人拿 mnist 或者 kaggle 数据集来练手，并且得出很好的结果，但是很少人拿训练模型去真实环境去测试验证，其正确性能有 80% 都不错了。为什么？不同地域，人们的书写习惯会不同，同时书写习惯也会随时间而改变，不同年龄段的人书写的规范程度也不一样，这些还只是真实环境错误预测的一小部分因素。现实中的人类可以根据数字所处的上下文来猜测模糊数字，或者不同格式的数字，例如 2^3，不会被认为是 2 和 3 而是 2 的立方。如果数字序列 3 5 7 9 中的 5 模糊掉了，那么人可以通过常识规律推测 5，而这种数学模型通过图像的特征进行识别就无能为力了。

所以人工智能在现实应用中既有非常大的限制，又有很大的用途。总结下来有几点：必须限制应用环境，复杂的应用环境准确性将严重下降，直至不可接受。其次必须是接受预测误差的应用场景，如果要求百分百准确，那么人工智能应用就只可以作为辅助（即便是作为辅助，它的威力依然惊人，如果在某种工作环节上它的准确性可以达到98%，那么这个工种环节就可以节约 98% 的人力费用，原来需要 100 个人的工作只需要 2 个人专门处理低置信度的未决预测就可以了，并且可以把这些错误预测收集归纳来训练新的模型，这样错误率就会越来越低，直至错误率低到无需人工干预也是可以接受的了）。 此外要认识到训练数据的准确性极其重要，不要期望通过调整模型来从不准确的数据中得出准确的预测结果。另外如果需要人工介入，就使用人工介入，人机交互中，人类具有一定的容忍度：比如谷歌搜索引擎会提示用户你要找是不是“xxx”，而不是在那里胡乱用复杂算法去猜测用户的想法，那样只会让体验愈加糟糕。

算法不能产生不存在的信息，Data talks。

迁移学习的思考
~~~~~~~~~~~~~~~~~

如果已经训练过一些模型，比如人脸识别，而要识别驴脸（纳尼，什么应用？），可能就麻烦了。人脸图片容易找，狗脸数据还能马马马虎凑合找到，更复杂的要识别驴脸麻烦就大了。另一特殊的样本采集起来可能非常麻烦，比如野生动物，或者特殊应用领域：微观领域（细胞，比如饮用水水质监测），宏观领域（航空，深空）。

还有上文的示例：现实中的人类可以根据数字所处的上下文来猜测模糊数字，或者不同格式的数字，例如 2^3，不会被认为是 2 和 3 而是 2 的立方。人类识别一样物品，例如狗狗，并不需要看太多狗的图片，而能从已有知识来加速学习：动物，有毛，四条腿，有尾巴，有耳朵，比马小，比猫大，叫起来汪汪。

迁移学习的本质就是基于已建立的深度神经网络模型对其中的部分层使用新数据集调节部分网络层权重（再训练）。这一技术从根本上解决了增量分类的重复训练问题。

Google 发布的 Inception 或 VGG16 这样成熟的物品分类的网络，只训练最后的 softmax 层，你只需要几千张图片，使用普通的 CPU 就能完成，而且模型的准确性不差。
Apple Turicreate 也是基于迁移学习，从而可以快速训练 CoreML 模型并部署到 iOS 上。

尽管如此，一堆所谓的有向无环图的“节点”（神圣地被称为“神经元”）组成的网络离真正意义上的“智能”还差得太远。

如果最终高效的人工智能算法模型被少数大公司垄断，只提供一些 API 接口（基本上这是一个趋势），那么人工智能的未来又该如何发展？

一些有趣的实践
~~~~~~~~~~~~~~~~~~

尽管机器学习和深度学习被大多应用于计算机视觉和自然语言(NLP)领域，但是如果把它放在其它领域其结果也会令人感到不可思议：

最近在从某网抽取数据来分析招聘信息，只从非常宏观的角度，就可以明显看出一个地区的产业分布（企业），人才层次分布，从这一分布就不难预测未来该地区的发展趋势。（政策层面如何量化？这确实是一个很大的变数，从各大官媒新闻报道中提及某些关键词频率入手？）。稍微细致分析，就可以看出某些公司的发展方向，人才储备的趋势变化。跟踪特定地区和公司的招聘变化相信将会有更大的发现。

再从雪球网抽取证券相关的评论信息（个人认为对于金融相关的预测过于关心过去的指数变化意义不大，反而可能从人的言行情绪上是一个不错的切入点），发现在负面情绪（负面分词占比很大）非常严重时，市场就开始具有不错的参与度（在不就的将来的收益很可能是超预期的），当然还要结合实际的宏观经济数据模型，不过至少它可以作为一个不错的特征指标，来衡量市场的冷热度。

当前阶段，人工智能领域最应该关注的趋势就是，算法模型向实际应用场景的落地。过多资源流向了算法研究，耗费在一堆参数上，而这些算法模型如何应用在各行各业，各个细分领域来产生实际的价值？

实战
------------------

令人印象“深刻”的示例
~~~~~~~~~~~~~~~~~~~~

人脸识别
``````````````````

有一次和一个朋友一起坐火车，入站的验票口不知被何时升级成了人脸自动识别系统，作为非计算机领域工作的朋友自然大为惊讶，一直在感叹世界变化太快！

opencv 源码中自带了一些人体识别的相关模型（人脸，身体或者眼球），它们位于 Library/etc/haarcascades 文件夹下，格式为 xml 文件。
haarcascade_frontalface_default.xml 就是较常使用的人脸识别模型之一。

.. code-block:: python
  :linenos:
  :lineno-start: 0

  # face_detect_haar.py

  # load opencv to handle image
  import cv2

  # load haar model and get face classifier
  faceModel = FaceDetector(r"models/haarcascades/haarcascade_frontalface_default.xml")
  faceClassifier = cv2.CascadeClassifier(faceModel)

  # load jpg file from disk
  image = cv2.imread("imgs/face.jpg")
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # get all faces returned in rects
  faceRects = faceClassifier.detectMultiScale(gray, 
                                              scaleFactor=1.5, 
                                              minNeighbors=5, 
                                              minSize=(30,30))
   
  print("I found %d face(s)" % (len(faceRects)))

  # draw rects on image and show up
  for x,y,w,h in faceRects:
      cv2.rectangle(image, (x,y), (x+w, y+h), (0, 255, 0), 2)
  cv2.imshow("Faces", image)
  cv2.waitKey(0)

短短几行代码就可以实现图片中人脸的识别：

- 首先导入 opencv，这里使用的版本为 4.0.1。这里 cv2 用于图片加载和保存，它是一个非常强大的图像处理库。
- 加载模型文件，并获取人脸分类器 faceClassifier。
- 从磁盘加载图片文件，由于 opencv 自带的人脸分类器只支持灰度图，这里先把 RGB 彩图转换为灰度图
- 使用分类器的 detectMultiScale 方法检测人脸，这里暂不讨论这些参数
- 打印识别到的人脸数目，同时在图像上绘制矩形并弹出显示窗口。

执行以上脚本：

.. code-block:: sh
  :linenos:
  :lineno-start: 0

  $ python face_detect_haar.py
  I found 2 face(s)

.. figure:: imgs/practice/face.png
  :scale: 100%
  :align: center
  :alt: face

  基于opencv模型的人脸识别

初次看到这类效果的人一定大为惊讶，并赞叹人工“智能”的神奇。

但是且慢，我们尝试对图片做一个最基本的缩放操作，再看看效果如何，为此我们增加一个缩放函数，并重新调整代码框架。

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  def img_resize(img, ratio=0.5, inter=cv2.INTER_AREA):
      w = img.shape[1] * ratio
      h = img.shape[0] * ratio
      
      return cv2.resize(img, (int(w), int(h)), interpolation=inter)

以上定义了一个缩放函数，ratio 指定了宽高缩放比，如果它小于1，图像将被缩小，否则将被放大。

接着定义处理参数的相关函数，以便传递参数：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  import cv2
  import argparse 
  
  def args_handle():
      ap = argparse.ArgumentParser()
      ap.add_argument("-i", "--image", required=False, 
                      default=r"imgs/face.jpg",
                      help="path to input image")
                    
      ap.add_argument("-m", "--model", required=False,
                      default=r"models/haarcascades/haarcascade_frontalface_default.xml",
                      help="path to opencv haar pre-trained model")
      
      return vars(ap.parse_args())
  
  g_args = None
  def arg_get(name): # 获取参数
      global g_args
      
      if g_args is None:
          g_args = args_handle()
      return g_args[name]

这里的参数列表只定义了名为 --image 和 --model 的两个参数，分别指定要进行人脸识别的图像路径和模型路径。接着封装一个用于人脸识别的 FaceDetector 类：

.. code-block:: python
  :linenos:
  :lineno-start: 0

  class FaceDetector():
      def __init__(self, model):
          self.faceClassifier = cv2.CascadeClassifier(model)
      
      # handle cv2 image object
      def detect_img(self, img, gray=1):  
          gray = img if gray == 1 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
          return self.faceClassifier.detectMultiScale(gray, 
                                                      scaleFactor=1.5, 
                                                      minNeighbors=5, 
                                                      minSize=(30,30))
      # handle image file
      def detect_fimg(self, fimg, verbose=0):
          # load jpg file from disk
          image = cv2.imread(fimg)
          gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
          
          faceRects = self.detect_img(gray, 1)
  
          # draw rects on image and show up
          for x,y,w,h in faceRects:
              cv2.rectangle(image, (x,y), (x+w, y+h), (0, 255, 0), 2) 
          
          return image
      
      def show_and_wait(self, image, title=' '):
          cv2.imshow(title, image)
          cv2.waitKey(0)

在 face_batchdetect_haar 中通过 img_resize 调整缩放比例从 10% 到 200% 以 10% 步长循环处理，然后对缩放过的图像进行人脸识别。

.. code-block:: python
  :linenos:
  :lineno-start: 0

  def face_batchdetect_haar_size(model_path, fimg):
      img = cv2.imread(fimg)
      FD = FaceDetector(model_path)
      for i in range(1, 21, 1):
          ratio = i * 0.1
          newimg = img_resize(img, ratio, inter=cv2.INTER_AREA)
          faceRects = FD.detect_img(newimg, gray=0)
          faces = len(faceRects)
          print("I found {} face(s) of ratio {:.2f} with shape{}".format(faces, 
                ratio, newimg.shape))
          for x,y,w,h in faceRects:
              cv2.rectangle(newimg, (x,y), (x+w, y+h), (0, 255, 0), 2)    
          if faces != 2 and faces != 0:
              FD.show_and_wait(newimg)
  
  model_path = arg_get('model')
  face_batchdetect_haar(model_path, 'imgs/face.jpg')

迫不及待等待结果。很可惜这个结果令人大跌眼镜，如果缩小图片另识别率降低可以情有可原（因为很小的图片，人眼也难以识别物体），竟然放大后的图片也会有问题，而且问题是各种各样，以示例图片的结果对此模型说明：

- 太小的分辨率无法识别图片，缩放到 20% 以下的图片已经无能为力
- 缩放到 50% 和 110% 的图片竟然能识别出 4 张人脸？
- 缩放到 80%，120%，160% 和 180% 的图片更神奇，识别出 3 张脸

不过可以看到图片的分辨率越小，越难以识别人脸，而不适当的分辨率也会导致识别出错，分辨率越大越不会丢失人脸，但是不要指望能保证正确率。

.. figure:: imgs/practice/err_faces.png
  :scale: 80%
  :align: center
  :alt: face

  基于opencv模型的人脸错误识别

.. code-block:: sh
  :linenos:
  :lineno-start: 0

  $ python face_detect_haar.py
  I found 0 face(s) of ratio 0.10 with shape(29, 60, 3)
  I found 0 face(s) of ratio 0.20 with shape(59, 120, 3)
  I found 2 face(s) of ratio 0.30 with shape(89, 180, 3)
  I found 2 face(s) of ratio 0.40 with shape(118, 240, 3)
  I found 4 face(s) of ratio 0.50 with shape(148, 300, 3)
  I found 2 face(s) of ratio 0.60 with shape(178, 360, 3)
  I found 2 face(s) of ratio 0.70 with shape(207, 420, 3)
  I found 3 face(s) of ratio 0.80 with shape(237, 480, 3)
  I found 2 face(s) of ratio 0.90 with shape(267, 540, 3)
  I found 2 face(s) of ratio 1.00 with shape(297, 600, 3)
  I found 4 face(s) of ratio 1.10 with shape(326, 660, 3)
  I found 3 face(s) of ratio 1.20 with shape(356, 720, 3)
  I found 2 face(s) of ratio 1.30 with shape(386, 780, 3)
  I found 2 face(s) of ratio 1.40 with shape(415, 840, 3)
  I found 2 face(s) of ratio 1.50 with shape(445, 900, 3)
  I found 3 face(s) of ratio 1.60 with shape(475, 960, 3)
  I found 4 face(s) of ratio 1.70 with shape(504, 1020, 3)
  I found 3 face(s) of ratio 1.80 with shape(534, 1080, 3)
  I found 2 face(s) of ratio 1.90 with shape(564, 1140, 3)
  I found 2 face(s) of ratio 2.00 with shape(594, 1200, 3)

到此我们对该模型的处理机制一无所知，它首先带来了惊喜，当然更多的是失望。这一模型被大家所诟病的问题不仅如此：它还会误识别，也即把根本不是人脸的图像识别为人脸；当人脸不是正面时，稍有角度不同识别率极度下降，正如模型的名称 frontalface 所讲。

不过从无到有总是困难的，这一模型至少说明人脸是可以通过计算机识别出来是可行的，而正确率是可以通过各种方式改善的。暂时忘记正确率吧，我们还可以在它上面继续挖掘一些有用的东西。

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  def face_batchdetect_haar(model_path, fimg):
      import time
      img = cv2.imread(fimg)
      FD = FaceDetector(model_path)
      for i in range(1, 21, 1):
          ratio = i * 0.1
          newimg = img_resize(img, ratio, inter=cv2.INTER_AREA)
          
          # time cost
          start = time.process_time()
          for i in range(0, 10):
              faceRects = FD.detect_img(newimg, gray=0)
          end = time.process_time()
          
          faces = len(faceRects)
          print("I found {} face(s) of ratio {:.2f} with shape{} cost time {:.2f}".format(faces, 
                ratio, newimg.shape, end - start))
          '''
          for x,y,w,h in faceRects:
              cv2.rectangle(newimg, (x,y), (x+w, y+h), (0, 255, 0), 2)    
          if faces != 2 and faces != 0:
              FD.show_and_wait(newimg, "{:.2f}".format(ratio))
          '''

以上代码对不同的图像大小统计人脸识别的耗时，这在实时处理的应用场景非要重要。对每种大小图片统计处理 10 次的时间：

.. code-block:: sh
  :linenos:
  :lineno-start: 0

  $ python face_detect_haar.py
  I found 0 face(s) of ratio 0.10 with shape(29, 60, 3) cost time 0.00
  I found 0 face(s) of ratio 0.20 with shape(59, 120, 3) cost time 0.03
  I found 2 face(s) of ratio 0.30 with shape(89, 180, 3) cost time 0.02
  I found 2 face(s) of ratio 0.40 with shape(118, 240, 3) cost time 0.11
  I found 4 face(s) of ratio 0.50 with shape(148, 300, 3) cost time 0.09
  I found 2 face(s) of ratio 0.60 with shape(178, 360, 3) cost time 0.33
  I found 2 face(s) of ratio 0.70 with shape(207, 420, 3) cost time 0.25
  I found 3 face(s) of ratio 0.80 with shape(237, 480, 3) cost time 0.12
  I found 2 face(s) of ratio 0.90 with shape(267, 540, 3) cost time 0.53
  I found 2 face(s) of ratio 1.00 with shape(297, 600, 3) cost time 0.62
  I found 4 face(s) of ratio 1.10 with shape(326, 660, 3) cost time 0.55
  I found 3 face(s) of ratio 1.20 with shape(356, 720, 3) cost time 0.86
  I found 2 face(s) of ratio 1.30 with shape(386, 780, 3) cost time 1.03
  I found 2 face(s) of ratio 1.40 with shape(415, 840, 3) cost time 0.84
  I found 2 face(s) of ratio 1.50 with shape(445, 900, 3) cost time 1.03
  I found 3 face(s) of ratio 1.60 with shape(475, 960, 3) cost time 1.14
  I found 4 face(s) of ratio 1.70 with shape(504, 1020, 3) cost time 1.41
  I found 3 face(s) of ratio 1.80 with shape(534, 1080, 3) cost time 1.58
  I found 2 face(s) of ratio 1.90 with shape(564, 1140, 3) cost time 1.64
  I found 2 face(s) of ratio 2.00 with shape(594, 1200, 3) cost time 1.80

上面的结果很令人满意：清楚的规律是，图像越大处理的耗时越长。笔者的笔记本 CPU 主频为 2.6GHz，常见的摄像头分辨率为 640*480，帧率 25-30，对应到上面的数据不难猜测大约为 1s，也即 1s 内处理 10 张 640*480 分辨率的图片，这似乎不是一个好消息。也即我们要丢到一半的帧率，如果对实时性要求很高，且不能丢帧，即便不从正确性上考虑，那么这个模型也有点悬。

如果要在嵌入式平台运行以上代码，并达到实时性要求，那么由于 ARM 之类的芯片主频没有笔记本主频这么高，那么就要考虑从硬件（DSP,FPGA,GPU）和软件(使用更高性能的编程语言/并行/图像缩小)两方面进行性能提升。

视频中识别人脸
```````````````

如果能从图片中识别出人脸，那么从视频数据中识别出人脸就不会很困难：由于视频流就是有多幅图片“组成的”，所以只要针对视频中的每一幅图片处理就可以达到目的了。

.. code-block:: python
  :linenos:
  :lineno-start: 0

  def face_detect_camera(model_path, show=0):
      import time
      frames = 0
      camera = cv2.VideoCapture(0)
      start = time.process_time()
      
      FD = FaceDetector(model_path)
      while(camera.isOpened()):
          grabbed, frame = camera.read()
          
          if not grabbed:
              print("grabbed nothing, just quit!")
              break
  
          faceRects = FD.detect_img(frame, gray=0)
          frames += 1
          
          fps = frames / (time.process_time() - start)
          print("{:.2f} FPS".format(fps), flush=True)
 
          if not show: # show video switcher
            contine
            
          cv2.putText(frame, "{:.2f} FPS".format(fps), (30, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

          cv2.imshow("Face", frame)          
          if cv2.waitKey(1) & 0xff == ord('q'):
              break

      camera.release()
      cv2.destroyAllWindows()
  
  model_path = arg_get('model')
  face_detect_camera(model_path)

我们从摄像头抓取视频帧，然后进行处理，首先跳过所有不必要的处理（这些处理我们可以放在其它线程或者进程中）：

.. code-block:: sh
  :linenos:
  :lineno-start: 0

  32.43 FPS
  32.00 FPS
  32.07 FPS
  32.14 FPS
  31.92 FPS
  ......

在最理想的情况下我们得到了以上结果，但是如果把笔记本的 2.6GHz 的算力换算到嵌入式平台上，情况依然不容乐观。到此为止我们打开视频流相关的代码，看看会发生什么：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  model_path = arg_get('model')
  face_detect_camera(model_path, show=1)

.. figure:: imgs/practice/video.png
  :scale: 50%
  :align: center
  :alt: face

  基于opencv模型的视频流人脸识别

帧率大约是 16 FPS，当然我们可以从软件层面挽回这一大约一倍的时间损失。

haar 模型的进一步思考
``````````````````````````

既然可以从图片尺寸和耗时上来考虑一个算法模型，那么我们不妨走得更远一些，看看会发生什么。

我们可以将图片围绕中心旋转，这是非常容易做到的。另外为了在旋转时头像始终处在图片之中，这里使用只有一张梦露脸的图片，且脸部基本位于图片中央。

.. figure:: imgs/practice/Monroe.jpg
  :scale: 80%
  :align: center
  :alt: Monroe

  图片旋转对人脸识别的影响用图

.. code-block:: python
  :linenos:
  :lineno-start: 0
    
  def rotate(image, angle):
      '''roate image around center of image'''
      h, w = image.shape[:2]
      center = (w // 2, h // 2)
      
      M = cv2.getRotationMatrix2D(center, angle, 1.0)
      return cv2.warpAffine(image, M, (w, h))
      
  # rotate a picture from 0-180 angle to check accuracy
  def face_batchdetect_haar_rotate(model_path, fimg):
      import time
      img = cv2.imread(fimg)
      FD = FaceDetector(model_path)
      for angle in range(0, 190, 10):
          newimg = rotate(img, angle)
          # time cost
          start = time.process_time()
          for i in range(0, 10):
              faceRects = FD.detect_img(newimg, gray=0)
          end = time.process_time()
          
          faces = len(faceRects)
          print("I found {} face(s) of rotate {} with shape{} cost time {:.2f}".format(faces, 
                angle, newimg.shape, end - start))
          
          for x,y,w,h in faceRects:
              cv2.rectangle(newimg, (x,y), (x+w, y+h), (0, 255, 0), 2)    
          if faces != 1 and faces != 0:
              FD.show_and_wait(newimg, "Rotate{}".format(angle))
          
  model_path = arg_get('model')
  face_batchdetect_haar_rotate(model_path, arg_get('image'))

结果令人大跌眼镜，旋转超过 10 度以后再难以识别出人脸，这令人不禁怀疑为何此模型的泛化能力如此之差？如果尝试在 -10到10度之间旋转，模型还是可以识别出人脸，这说明模型在训练之初使用的数据很可能没有考虑这种特殊情况。 

.. code-block:: sh
  :linenos:
  :lineno-start: 0
  
  $ python face_detect_haar.py  -i imgs/Monroe.jpg
  I found 1 face(s) of rotate 0 with shape(480, 640, 3) cost time 0.84
  I found 0 face(s) of rotate 10 with shape(480, 640, 3) cost time 0.48
  I found 0 face(s) of rotate 20 with shape(480, 640, 3) cost time 0.61
  ......
  I found 0 face(s) of rotate 130 with shape(480, 640, 3) cost time 0.53
  I found 1 face(s) of rotate 140 with shape(480, 640, 3) cost time 0.53
  I found 0 face(s) of rotate 150 with shape(480, 640, 3) cost time 0.50
  I found 0 face(s) of rotate 160 with shape(480, 640, 3) cost time 0.67
  I found 0 face(s) of rotate 170 with shape(480, 640, 3) cost time 0.52
  I found 0 face(s) of rotate 180 with shape(480, 640, 3) cost time 0.73

如果我们只是对图片进行水平和垂直方向的平移，那么识别率会怎么变化？理论上应该不会有影响。事实却非如此。

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  def translation(image, x, y):
      '''move image at x-axis x pixels and y-axis y pixels'''
      
      M = np.float32([[1, 0, x], [0, 1, y]])
      return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
  
  def face_batchdetect_haar_move(model_path, fimg):
      import time
      img = cv2.imread(fimg)
      FD = FaceDetector(model_path)
      for move in range(0, 100, 10):
          newimg = translation(img, move, move)
          # time cost
          start = time.process_time()
          for i in range(0, 10):
              faceRects = FD.detect_img(newimg, gray=0)
          end = time.process_time()
          
          faces = len(faceRects)
          print("I found {} face(s) of move {} with shape{} cost time {:.2f}".format(faces, 
                move, newimg.shape, end - start))
          
          for x,y,w,h in faceRects:
              cv2.rectangle(newimg, (x,y), (x+w, y+h), (0, 255, 0), 2)    
          #if faces != 1 and faces != 0:
          FD.show_and_wait(newimg, "Move{}".format(move))
  
  model_path = arg_get('model')
  face_batchdetect_haar_move(model_path, arg_get('image'))

结果还是令人大跌眼镜，将图像向右下方以 10 像素每步移动，有时可以识别，有时失败，毫无规律可循。这说明此模型对背景敏感，由于我们在旋转和平移时背景均被填充为了黑色，这与原图的背景色并不完全一致。笔者尝试在识别前进行高斯模糊，效果就出现了改善。

.. code-block:: sh
  :linenos:
  :lineno-start: 0
  
  $ python face_detect_haar.py  -i imgs/Monroe.jpg
  I found 1 face(s) of move 0 with shape(480, 640, 3) cost time 0.66
  I found 1 face(s) of move 10 with shape(480, 640, 3) cost time 0.48
  I found 0 face(s) of move 20 with shape(480, 640, 3) cost time 0.53
  I found 1 face(s) of move 30 with shape(480, 640, 3) cost time 0.52
  I found 1 face(s) of move 40 with shape(480, 640, 3) cost time 0.48
  I found 0 face(s) of move 50 with shape(480, 640, 3) cost time 0.45
  I found 0 face(s) of move 60 with shape(480, 640, 3) cost time 0.64
  I found 1 face(s) of move 70 with shape(480, 640, 3) cost time 0.56
  I found 0 face(s) of move 80 with shape(480, 640, 3) cost time 0.55
  I found 0 face(s) of move 90 with shape(480, 640, 3) cost time 0.52

经历了漫长的测试验证，我们将该模型最为黑盒使用，依然对模型本身不甚了解，但是至少可以知道不要轻易对一个看起来令人“惊喜”的模型太过乐观，对它们的使用常常是有严格限制条件的。好吧，就从这里开始人工智能的实战之路。

距离和kNN分类
~~~~~~~~~~~~~~

勾股定理（毕达哥拉斯定理）是数学史上最伟大定理之一，除了因为它引入了无理数，还因为它使得几何距离在坐标中可以计算，它把坐标张开成面和3维空间，甚至高维空间。

人类生活的3维世界被形形色色的物体充满，有些还无色无味，为了描述这些物体，区分和应用，从感官层面人类发展出各类描述词汇，形状，颜色，味道，密度，重量等等。

所有事物似乎都可以用一棵树一样的形状进行从粗到细的分类，比如生物学上的界门科目属种。离根越近的分类，它们的共同点就越接近本质，而离树梢越近的分类就只有细微的区别，同一个末梢上的分支也就具有更多的相同特征，比如哈士奇和萨摩耶。人类在描述相近事物时彼此已经建立了共同的理解基础，所以只要说是犬类，大家都明白毛茸茸，有四条腿，有耳朵，有尾巴，叫起来汪汪。没有人会描述这些共同的特征来介绍一只狗，而是直接说出区别于其他犬种的细节，比如体型小，善狩猎等等。

我们不想一开始就区分两种犬类的图片，而是从更少特征值的区分上进行入手。

考虑数字 1 和 2，以及 10000，我们自然认为 1 和 2 非常接近，但是计算机没有这种感觉，它没法感觉远近，只不过是内存中存储的二进制而已。在计算机中所有的数据都是二进制数据，要感知距离就需要给计算机规则，从计算上来区分距离。

从主观猜测开始
```````````````

计算机中的数与数之间的距离可以用减法定义，而一组数和另一组数之间的距离就可以用向量距离来定义（这就用到了勾股定理）。一张图片就是一组像素值，是否可以把像素值直接展成一维向量，来计算它们之间的距离，如果对两张复杂图片适用，那么对于最简单的二值图像更会适用。这里不妨拿出最简的四个像素来组成一幅二值图。

只有 4 个像素的二值图图片依然可以表达非常丰富的信息，因为有 2^4 = 16 种组合。可以想见人们在一个 20*20 的像素方格内书写 0-9，相对于整个组合的情况是多么地稀疏。我们只使用了像素空间的很小部分，以便于人眼的识别，所以这里我们使用四个像素生成 3 幅图，分别对应符号 "\-\|\_"，这对于人眼一目了然。

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  import numpy as np
  import cv2
  
  def bitwise(imga, imgb=None, opt='not'):
      '''bitwise: and or xor and not'''
      if opt != 'not' and imga.shape != imgb.shape:
          print("Imgs with different shape, can't do bitwise!")
          return None
  
      opt = opt.lower()[0]
      if opt == 'a':
          return cv2.bitwise_and(imga, imgb)
      elif opt == 'o':        
          return cv2.bitwise_or(imga, imgb)
      elif opt == 'x':
          return cv2.bitwise_xor(imga, imgb)
      elif opt == 'n':
          return cv2.bitwise_not(imga)
  
      print("Unknown bitwise opt %s!" % opt)
      return None
  
  def vector_dist(V0, V1):
      from numpy import linalg as la
      V0 = np.array(V0).astype('float64')
      V1 = np.array(V1).astype('float64')
  
      return la.norm(V1 - V0)
  
  def show_simple_distance():
      gray0 = np.array([[0,0],[255,255]], dtype=np.uint8)
      gray1 = gray0.transpose()
      
      cv2.imshow('-', gray0)
      cv2.imshow('|', gray1)
      
      gray2 = bitwise(gray0, None, opt='not')
      cv2.imshow('_', gray2)
      
      g01 = vector_dist(gray0, gray1)
      g02 = vector_dist(gray0, gray2)
      print("distance between -| is {}, distance between -_ is {}".format(int(g01), int(g02)))
      
      cv2.waitKey(0)
  
  show_simple_distance()

.. figure:: imgs/practice/dist.png
  :scale: 80%
  :align: center
  :alt: Monroe

  四像素的二值图

四像素的二值图无法表示复杂的数字形状，但是可以表示一横和一竖，从这个角度看左边两幅图应该距离更近，上边的两幅图应该距离更远，然而通过展开 2*2 的四像素成为 4 维向量，然后求取它们的向量距离：

.. code-block:: sh
  :linenos:
  :lineno-start: 0
  
  $ python vector_distance.py
  distance between -| is 360, distance between -_ is 510
  
显然左边两幅图距离为 510，比上边的两幅图距离更远，这不是我们所期待的，难道通过这种向量方式的距离求取来分类像素组成的几何形状根本不可行？

在人类的世界里面不存在任何像素，而只有事物映射到大脑的信息：大小，形状，颜色。如果看到一个数字，基于过往的视觉经验，首先人脑会不自主得进行中心视觉的处理：如果两个数字是黏连的，人脑会主动分割；如果数字是模糊的人脑也会根据边界自动区分；如果数字是歪斜的，甚至颠倒的，人脑会自动纠正（过滤干扰）。人脑对每一个数字形成一个完整的标准的数字形象，当视觉神经细胞接收一个类似数字的符号后，人脑自动与标准数字形象进行比较，哪个最相像，哪一个就是要识别的数字。

这一过程，计算机是完全无知的，但是可以从算法上模拟。如果只有 4 个像素，那么考虑“中心视觉”就不现实了，这犹如人眼盯着放大数字的一角。在一个 20*20 的像素空间内计算机就可以形成“中心视觉”了（此时的向量距离就能反馈数字相似性的信息），例如 1 的像素值总是集中在 7-12 列上，且前几行和后几行像素通常都是空白的。

mnist 数据集上的试验
```````````````````````

这里借用 mnist 手写数据集，每个数字由 28*28 个像素组成。

.. code-block:: python
  :linenos:
  :lineno-start: 0
    
  import dbload
  
  # imgs with shape(count,height,width)
  def show_gray_imgs(imgs, title=' '):
      newimg = imgs[0]
      for i in imgs[1:]:
          newimg = np.hstack((newimg, i))
      
      cv2.imshow(title, newimg)
      cv2.waitKey(0)
  
  train,labels = dbload.load_mnist(r"./db/mnist", kind='train', count=20)
  num1 = train[labels==1]
  print(len(num1))
  
  show_gray_imgs(num1, '1')
  
  >>>
  4

首先读取训练集中的前 20 个样本，然后取数数字 1，可以看到有 4 个数字 1 被取出，打印出来看看效果：

.. figure:: imgs/practice/41.png
  :scale: 80%
  :align: center
  :alt: Monroe

  手写数字 1

尽管对于人脑来说上面的数字（除非不限定在数字范围来考虑这些符号）一目了然，并且可以轻易的得出这四个1之间的“距离”（相似度），第一个 1 向左倾斜一个很大角度，和其他三个 1 距离最远，最后两个 1 之间距离最小。如果把问题聚焦在第一个1和其余三个1的距离比较上，显然距离第二个1距离最大，距离最后边的两个1距离差不多：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  for i in range(1, len(num1)):
      print("distance between 0-{} {}".format(i, vector_dist(num1[0], num1[i])))
     
  >>>
  distance between 0-1 2354.3323894471655
  distance between 0-2 2152.188885762586
  distance between 0-3 2114.714401520924

结果和我们的预测如此吻合，很令人惊讶。如果第一个1是靠近左上角，或者右下角，或者某一侧，那么计算机就无法再形成“中心视觉”了，可以想见它距离中心视觉的1的距离就会很远。如何克服这一问题？符号处于空间的位置不影响人脑识别出这一符号，也即人脑能很好得过滤这些干扰，计算机无法自动识别（在这一简单的距离模型下）这一干扰，需要人为来构造建立“中心视觉”的环境。

可以想见这一“环境”是怎样的————令待识别的图像最接近理想的标准的数字形象：

- 位置：数字位置应该处于图像中心，以最完整的方式清晰展现出来
- 角度：数字不应该有较大的倾斜角度，而是端端正正的
- 扭曲：数字不应该有较大的扭曲，比如 1 应该是一条直线，而不是竖起来的波浪线
- 大小：数字所占的整个比例应该和整个画布比例一致，不应该太小或太大
- 亮度：对于灰度图，需要考虑亮度的影响，而对于二值图就可以忽略虑亮度的影响

尽管还有一些其它的次要因素，比如边缘应该平滑无毛刺，但这些不是主要因素。事实上 mnist 数据集在采集时已经做了这些处理，每一个数字看起来都能很好得获取到“中心视觉”。这也就是为何 mnist 数据集在很多简单的模型上都能获取很高的识别率的重要因素，如果使用这些模型来验证其他渠道采集来的数字图像，并且这些数字图像不进行以上处理，结果就会令人大跌眼镜。

我们继续验证第一个数字 1 和其他数字的距离：

.. code-block:: python
  :linenos:
  :lineno-start: 0

  for i in range(1, len(train)):
      print("distance between 0-{} {}".format(labels[i], vector_dist(num1[0], train[i])))
  
  >>>
  distance between 0-1 0.0
  distance between 0-9 2388.816652654615
  distance between 0-2 2525.059603256921
  distance between 0-1 2354.3323894471655
  distance between 0-3 2604.63471527199
  distance between 0-1 2152.188885762586
  distance between 0-4 2397.628203037327
  distance between 0-3 2499.4817462826168
  distance between 0-5 1916.8805387921282
  distance between 0-3 2850.328402131937
  distance between 0-6 2611.602190227294
  distance between 0-1 2114.714401520924
  distance between 0-7 2411.6311907088943
  distance between 0-2 2491.427703145327
  distance between 0-8 1914.8302796853825
  distance between 0-6 2259.1578076796673
  distance between 0-9 2019.5298957925827

这里的 0-x 中的 x 不再是其他 1 的索引，而是换成了数字的下标。这里与训练集中的 20 个数字进行了距离计算。

很容易看出来，1 与 其他数字的距离都比较远，离其他 1 距离较近。此时不难想出一个简单的数字分类算法：在样本上计算距离，找出最近的几个样本，查看它们的标签，最多标签标示的数字的就是最可能的数字。

注意：此时的计算机无法识别大角度旋转甚至倒立的数字，这需要数据的预处理。
 
kNN 邻近算法
`````````````

K 最近邻(kNN，k-NearestNeighbor)分类算法是数据挖掘分类技术中最简单的方法之一。相对于其他复杂的多参数机器学习模型，它非常简单，无需学习，直接通过强力计算来进行分类。

上一节已经揭示了 K 最邻近算法的本质：计算与已知样本的距离，选取 k 个距离最小（最邻近）的样本，统计这些最邻近样本的标签，占比最大的标签就是预期值。显然最邻近的 k 个样本具有投票权，哪种标签票数多，哪种标签就获胜。

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  # knn_mnist.py
  def kNN_predict(train, labels, sample, k=5):
      import operator
      
      # 使用矩阵方式计算 sample 和训练集上的每一样本的向量距离
      diff = train.astype('float64') - sample.astype('float64')
      distance = np.sum(diff ** 2, axis=2)
      distance = np.sum(distance, axis=1) ** 0.5
      
      # 对向量距离排序，获取排序索引，进而获取排序标签
      I = np.argsort(distance)
      labels = labels[I]
      
      max_labels = {}
      if len(train) < k:
          k = len(train)
      
      # 统计前 k 个投票的标签信息
      for i in range(0,k):
          max_labels[labels[i]] = max_labels.get(labels[i], 0) + 1
    
      # 返回从大到小票数排序的元组
      return sorted(max_labels.items(), key=operator.itemgetter(1), reverse=True)

kNN 算法实现非常简单，计算待预测样本与训练集上每一样本的向量距离，提取前 k 个距离最近的标签信息，统计标签列表，返回从大到小票数排序的元组。

从程序实现上可以感觉到，kNN 的计算非常耗时，训练集越大，计算量将线性增加，当然这可以通过多线程/进程采用分治法降低计算复杂度；但是另一个问题却无法解决，算法对磁盘空间和内存空间的占用。训练集越大，占用的磁盘空间和内存空间就越大，如果采用缓存方式就牺牲了计算性能。

实际验证可以发现，kNN 算法的效果非常好，可以轻易达到 98% 以上的准确度，且无需训练。当然准确度依赖性也很强，采用的训练集的样本数和分布，k 值的选择都对结果有影响。可以通过交叉验证来选择一个比较优的 k 值，默认值是5。

.. code-block:: python
  :linenos:
  :lineno-start: 0

  def kNN_test(train_entries=10000, test_entries=10000):
      k = 5
  
      train,labels = dbload.load_mnist(r"./db/mnist", kind='train', count=train_entries)
      test,test_labels = dbload.load_mnist(r"./db/mnist", kind='test', count=test_entries)
  
      error_entries = 0
      start = time.process_time()
      for i in range(0, test_entries):
          max_labels = kNN_predict(train, labels, test[i], k=k)
          predict = max_labels[0][0]
          if(predict != test_labels[i]):
              error_entries += 1
              #print(predict, test_labels[i], flush=True)
              #cv2.imshow("Predict:{} Label:{}".format(predict, test_labels[i]), test[i])
  
      print("Average cost time {:.02f}ms accuracy rate {:.02f}% on trainset {}".format(
            (time.process_time() - start) / test_entries * 1000,
            (test_entries - error_entries) / test_entries * 100,
            train_entries), flush=True)
      #cv2.waitKey(0)
  
  def kNN_batch_test():
      for i in range(10000, 70000, 10000):
          print("trains {}".format(i), flush=True)
          kNN_test(i, 1000)

采用批量方式在测试集上验证 1000 个样本，训练集从 10000-60000 以 10000 步递进：

.. code-block:: sh
  :linenos:
  :lineno-start: 0
  
  $ python knn_mnist.py
  Average cost time 135.38ms accuracy rate 92.00% on trainset 10000
  Average cost time 283.84ms accuracy rate 93.80% on trainset 20000
  Average cost time 417.44ms accuracy rate 94.40% on trainset 30000
  Average cost time 575.08ms accuracy rate 96.30% on trainset 40000
  Average cost time 722.20ms accuracy rate 98.00% on trainset 50000
  Average cost time 847.16ms accuracy rate 98.20% on trainset 60000

从结果上不难看出，数字识别平均耗时，与训练集的大小成线性增加，准确度在达到一定程度后就难以提升，但是输出预测结果很稳定，我们可以查看这些识别错误的字符，来分析一下可能性：两个数字看起来很像，体现在像素分布上应该差不多。

.. figure:: imgs/practice/err_num.png
  :scale: 80%
  :align: center
  :alt: face

  kNN 算法识别错误的数字示例

观察这些被错误识别的数字很有趣。我们可以把错误情况分为两类：

- 情有可原的一类，这类数字即便人工也难以辨别。上面的大部分情况属于这类。如果要对这类数字进行优化，可以想见将影响其他已经正确识别的数字的正确率。
- 右下角的 6 尽管书写很不规范，但是人脑很容易就识别出来，算法将它识别为 1， 显然是符合像素组成的向量距离最优的，但是这种最优和人脑识别数字的准确性出现了明显偏差。

经过以上分析，可能会意识到，人脑识别数字并不是靠像素构成的向量距离来判断相似性这么简单，而是使用更深层次的特征。人类认识 0-9 个符号，不需要看大量的图片，也不需要进行大量计算，而是会在大脑中形成标准的数字图像符号，此外人脑具有很行的过滤干扰的能力。这一切“智能”都是朴素的 kNN 算法所不具备的。

scikit-learn kNN算法
`````````````````````

scikit-learn 模块实现了传统机器学习的各类算法，并进行了大量优化，借此无需在制造不好用的轮子。这里对 scikit-learn kNN算法进行定量的性能分析。

.. code-block:: python
  :linenos:
  :lineno-start: 0

  def kNN_sklearn_predict(train, labels, test):
      from sklearn.neighbors import KNeighborsClassifier
      knn = KNeighborsClassifier()
      knn.fit(train, labels)
  
      return knn.predict(test)
      
  def kNN_sklearn_test(train_entries=10000, test_entries=1000):  
      train,labels = dbload.load_mnist(r"./db/mnist", kind='train', count=train_entries)
      test,test_labels = dbload.load_mnist(r"./db/mnist", kind='test', count=test_entries)
      
      train = train.reshape((train_entries, train.shape[1] * train.shape[2]))
      test = test.reshape((test_entries, test.shape[1] * test.shape[2]))
      
      start = time.process_time()
      predict = kNN_sklearn_predict(train, labels, test)
      error = predict - test_labels
      error_entries = np.count_nonzero(error != 0)
  
      print("Average cost time {:.02f}ms accuracy rate {:.02f}% on trainset {}".format(
            (time.process_time() - start) / test_entries * 1000,
            (test_entries - error_entries) / test_entries * 100,
            train_entries), flush=True)
  
  def kNN_sklearn_batch_test():
      for i in range(10000, 70000, 10000):
          kNN_sklearn_test(i, 1000)
          
  kNN_sklearn_batch_test()

采用同样的批量测试方法，来对比 scikit-learn 封装的 kNN 算法的性能，需要注意到 scikit-learn 对 kNN 算法进行了大量的技巧性的扩展：

- 距离度量 metric ：通常使用欧氏距离，默认的 minkowski 距离在 p=2 时就是欧氏距离
- algorithm ：4 种可选，‘brute’对应蛮力计算，‘kd_tree’对应 KD树 实现，‘ball_tree’ 对应球树实现， ‘auto’则会在上面三种算法中做权衡，选择一个拟合最好的最优算法。需要注意的是，如果输入样本特征是稀疏的时候，无论我们选择哪种算法，最后scikit-learn都会去用蛮力实现‘brute’。
- 并且处理任务书 n_jobs：用于多核CPU时的并行处理，加快建立KNN树和预测搜索的速度。一般用默认的 -1 就可以了，即所有的CPU核都参与计算。
- n_neighbors：最近邻个数，通常选择默认值 5。
- 近邻权 weights ：'uniform' 意味着最近邻投票权重均等。"distance"，则权重和距离成反比例，即距离预测目标更近的近邻具有更高的权重，更近的近邻所占的影响因子会更加大。 

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  # 默认 scikit-learn 封装的 kNN 算法参数
  KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                       metric_params=None, n_jobs=1, n_neighbors=5, p=2,
                       weights='uniform')

scikit-learn 封装的 kNN 算法计算速度有了很大的提升，比自实现算法速度快大约 7-8 倍。准确率上有所降低，但基本不相上下。

.. code-block:: sh
  :linenos:
  :lineno-start: 0
  
  $ python knn_mnist.py
  Average cost time 18.14ms accuracy rate 91.60% on trainset 10000
  Average cost time 36.64ms accuracy rate 93.70% on trainset 20000
  Average cost time 51.38ms accuracy rate 94.70% on trainset 30000
  Average cost time 77.83ms accuracy rate 96.00% on trainset 40000
  Average cost time 93.25ms accuracy rate 95.70% on trainset 50000
  Average cost time 109.94ms accuracy rate 96.10% on trainset 60000

kNN 并行参数
`````````````

在以上的各类参数中，有一个很吸引人的参数 n_jobs，它的默认值为 1，只使用了一个 CPU 核，在多核心的CPU上，这个参数对性能影响巨大。scikit-learn 并行操作使用 Joblib 的 Parallel 类实现。当笔者打开该参数时，发现性能不仅没有提升还略有降低，实际上是统计时间的代码问题。

time.process_time() 方法返回本进程或者线程的所有 CPU 核的占用时间，包括用户时间和系统时间，不包含 sleep 时间。所以算上启动多进程，以及数据多核心的分割和结果合并处理时间，占用的所有 CPU 核的时间就会略有上升。该函数对于性能瓶颈分析很有用。

统计相对于真实世界的耗时可以采用墙上时间函数 time.time()，修改代码如下：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  def kNN_sklearn_test(train_entries=10000, test_entries=1000):  
      train,labels = dbload.load_mnist(r"./db/mnist", kind='train', count=train_entries)
      test,test_labels = dbload.load_mnist(r"./db/mnist", kind='test', count=test_entries)
      
      train = train.reshape((train_entries, train.shape[1] * train.shape[2]))
      test = test.reshape((test_entries, test.shape[1] * test.shape[2]))
      
      stime = time.process_time()
      wstime = time.time()        # 显示墙上时间
  
      predict = kNN_sklearn_predict(train, labels, test)
      error = predict.astype(np.int32) - test_labels.astype(np.int32)
      error_entries = np.count_nonzero(error != 0)
  
      print("Average cost cpu time {:.02f}ms walltime {:.02f}s"
            " accuracy rate {:.02f}% on trainset {}".format(
            (time.process_time() - stime) / test_entries * 1000,
            (time.time() - wstime),
            (test_entries - error_entries) / test_entries * 100,
            train_entries), flush=True)
  
  # Joblib 启动多线程时会检查脚本是否为主程序调用
  if __name__ == '__main__':
      kNN_sklearn_batch_test()

n_jobs = -1 使用所有核，可以通过 Windows 资源监视器查看 CPU 使用情况。

.. code-block:: sh
  :linenos:
  :lineno-start: 0
  
  # n_jobs = 1 时使用一个 CPU 核
  $ python knn_mnist.py
  Average cost cpu time 17.94ms walltime 17.97s accuracy rate 91.60% on trainset 10000
  Average cost cpu time 36.03ms walltime 36.08s accuracy rate 93.70% on trainset 20000
  Average cost cpu time 50.50ms walltime 50.52s accuracy rate 94.70% on trainset 30000
  Average cost cpu time 76.39ms walltime 76.51s accuracy rate 96.00% on trainset 40000
  Average cost cpu time 99.47ms walltime 99.74s accuracy rate 95.70% on trainset 50000
  Average cost cpu time 115.23ms walltime 115.41s accuracy rate 96.10% on trainset 60000
  
  # n_jobs = -1 使用所有核，笔者环境为 8 核心
  $ python knn_mnist.py
  Average cost cpu time 22.64ms walltime 4.58s accuracy rate 91.60% on trainset 10000
  Average cost cpu time 47.11ms walltime 10.15s accuracy rate 93.70% on trainset 20000
  Average cost cpu time 67.48ms walltime 16.25s accuracy rate 94.70% on trainset 30000
  Average cost cpu time 96.39ms walltime 23.10s accuracy rate 96.00% on trainset 40000
  Average cost cpu time 119.05ms walltime 30.40s accuracy rate 95.70% on trainset 50000
  Average cost cpu time 144.48ms walltime 41.26s accuracy rate 96.10% on trainset 60000

对比以上两组数据，可以非常清晰地看到，墙上时间（现实世界中的耗时）明显降低，大约降低了 3 倍。n_jobs 参数在多核环境是非常有效的提速工具。

kNN 近邻权参数
`````````````````

另一个令人关注的参数是近邻权 weights。思考待识别样本距离更近的样本点的投票权重更大，而不是简单的取平均，将会校正这样一个错误：由于书写的扭曲，模糊，等等不规范问题导致某个数字应该分布在距离很近的一个范围内，可以想象成大部分样本点聚集在一个圆内，现在某个待测样本落在了圆外，并且靠近（还未落入）了另外一个数字聚集的圆，这个圆内有很多样本具有了表决权，如何才能把它拉回正确的圆内？

显然只能增加正确的少数派的投票权重，当然这是一种人为干预：主观认为距离越近就越加相似（这也是 kNN 算法的思想，既然整体上是对的，那么它在细节上应该也是对的）。

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  def kNN_sklearn_predict(train, labels, test):
      from sklearn.neighbors import KNeighborsClassifier
      knn = KNeighborsClassifier(algorithm='auto', n_jobs=-1, weights='distance')
      knn.fit(train, labels)
  
      return knn.predict(test)

更新 kNN_sklearn_predict 函数，设置 weights 参数为 distance。来看一下效果，大约有 0.2%-0.4% 的微弱提升。

.. code-block:: sh
  :linenos:
  :lineno-start: 0
  
  $ python knn_mnist.py
  Average cost cpu time 22.12ms walltime 4.40s accuracy rate 91.90% on trainset 10000
  Average cost cpu time 44.66ms walltime 9.03s accuracy rate 93.80% on trainset 20000
  Average cost cpu time 65.02ms walltime 14.29s accuracy rate 94.50% on trainset 30000
  Average cost cpu time 94.42ms walltime 22.28s accuracy rate 96.30% on trainset 40000
  Average cost cpu time 118.08ms walltime 30.93s accuracy rate 96.30% on trainset 50000
  Average cost cpu time 142.30ms walltime 41.20s accuracy rate 96.40% on trainset 60000

算法特征
`````````````````

蛮力计算(brute)：计算预测样本和所有训练集中的样本的距离，然后计算出最小的k个距离即可，接着多数表决。这个方法简单直接，在样本量少，样本特征少的时候很有效。比较适合于少量样本的简单模型的时候用。

brute 算法在 mnist 数据集上，速度很快：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  def kNN_sklearn_predict(train, labels, test):
      from sklearn.neighbors import KNeighborsClassifier
      knn = KNeighborsClassifier(algorithm='brute', n_jobs=-1)
      knn.fit(train, labels)
  
      return knn.predict(test)
      
.. code-block:: sh
  :linenos:
  :lineno-start: 0

  $ python knn_mnist.py
  Average cost cpu time 1.08ms walltime 7.77s accuracy rate 91.60% on trainset 10000
  Average cost cpu time 1.05ms walltime 5.62s accuracy rate 93.70% on trainset 20000
  Average cost cpu time 1.50ms walltime 4.61s accuracy rate 94.70% on trainset 30000
  Average cost cpu time 2.28ms walltime 18.43s accuracy rate 96.00% on trainset 40000
  Average cost cpu time 2.94ms walltime 22.10s accuracy rate 95.70% on trainset 50000
  Average cost cpu time 3.73ms walltime 16.73s accuracy rate 96.10% on trainset 60000

KD树（k-dimensional树的简称），是一种分割 k 维数据空间的数据结构，主要应用于多维空间关键数据的近邻查找(Nearest Neighbor)和近似最近邻查找(Approximate Nearest Neighbor)。本质上 KD 树就是二叉查找树（Binary Search Tree，BST）的变种。KD树实现和球树实现原理大体相同，均是对数据进行预分类。

更改参数 algorithm 分别为 "kd_tree" 和 "ball_tree"，以下是两种算法的效果对比，两者的预测准确率完全一致（在 mnist 数据集上），ball_tree 算法速度稍快：

.. code-block:: sh
  :linenos:
  :lineno-start: 0

  # kd_tree 算法效果
  Average cost cpu time 22.58ms walltime 4.36s accuracy rate 91.60% on trainset 10000
  Average cost cpu time 44.25ms walltime 8.23s accuracy rate 93.70% on trainset 20000
  Average cost cpu time 64.89ms walltime 13.19s accuracy rate 94.70% on trainset 30000
  Average cost cpu time 96.47ms walltime 22.21s accuracy rate 96.00% on trainset 40000
  Average cost cpu time 120.00ms walltime 28.58s accuracy rate 95.70% on trainset 50000
  Average cost cpu time 143.03ms walltime 37.58s accuracy rate 96.10% on trainset 60000

  # ball_tree 算法效果
  Average cost cpu time 17.91ms walltime 3.65s accuracy rate 91.60% on trainset 10000
  Average cost cpu time 38.00ms walltime 7.35s accuracy rate 93.70% on trainset 20000
  Average cost cpu time 59.30ms walltime 12.50s accuracy rate 94.70% on trainset 30000
  Average cost cpu time 84.50ms walltime 21.21s accuracy rate 96.00% on trainset 40000
  Average cost cpu time 110.95ms walltime 29.79s accuracy rate 95.70% on trainset 50000
  Average cost cpu time 133.73ms walltime 37.34s accuracy rate 96.10% on trainset 60000

kNN 算法启示
`````````````

下图可以看出错误率（评估算法准确性常用这一指标）随着训练集的样本的增大，在不停降低，但是下降速度越来越慢：

.. figure:: imgs/practice/knn_err_ratio.png
  :scale: 100%
  :align: center
  :alt: knn_err_ratio

  错误率和样本数关系曲线

为何下降速度越来越慢，一个启发性解释：训练样本的像素的向量终点在高维空间落在不同的区域，相同数字的向量终点回聚集在一个小的范围内（距离近，夹角小），这一范围如果映射成到平面上，就可以想象成一个圆形（当然也可以是其他可以描述一片聚集区域的图形）区域，越靠近圆心训练样本越密集，越靠近边界分布越稀少（如果从像素的直方图上统计相同数字的分布符合正态分布，那么映射到高维空间不会改变这一分布特性）。当训练样本很少时，这个圆的形状就不能完全体现出来，当样本越多，那么这个圆形就很越加完美的显现出来，当到达一定程度后，更密集的训练样本就很难对圆形的表达力进行提高了。

使用正态分布（高斯分布）来模拟这种情况：

.. code-block:: python
  :linenos:
  :lineno-start: 0
  
  def draw_normal_distribution(points=100):
      import matplotlib.pyplot as plt
  
      np.random.seed(0)
      rand_num = np.random.normal(0, 1, (4, points))
      Ax, Ay = rand_num[0] - 3, rand_num[1] - 3
      Bx, By = rand_num[2] + 3, rand_num[3] + 3
       
      plt.figure()
      plt.title("Normal Distribution with {} points".format(points))
      plt.xlim(-10, 10) 
      plt.ylim(-10, 10) 
  
      plt.scatter(Ax, Ay, s=5, c='black')
      plt.scatter(Bx, By, s=5, c='black')
      plt.show()

这里为了模拟分类，分别绘制两个点聚集的区域：

.. figure:: imgs/practice/100.png
  :scale: 100%
  :align: center
  :alt: knn_err_ratio

  绘制 100 个正态分布点

当样本点比较少的时候，我们不易观察出这种分布的聚集规律，当样本点从100个增大100倍到10000个点时，就非常显著了：

.. figure:: imgs/practice/10000.png
  :scale: 100%
  :align: center
  :alt: knn_err_ratio

  绘制 10000 个正态分布点

通常人书写时有某种倾向，比如向左倾斜，那么图形看起来就不会是正圆，就会被拉长成椭圆，当然其他倾向会对聚集的空间形状也有扭曲影响。如果我们把这种人书写的各种倾向进行泛化，比如对图片统一进行左倾，右倾，或者扭曲，抖动处理，那么这个圆形就接近正圆（这里看起来是椭圆，是因为图片长宽高比例不同）了。（这里假设人手写数字符合正态分布，当然也可以是其他分布，只是形状不同）。

经过优化的算法库的性能要远远优于未优化的代码，尝试不同软件包提供的同种算法，会发现性能上有很大区别。

另外从矩阵计算向量距离的方式上可以看到，使用任何一种方式把图像向量化（二维变一维）都是等价的，无论是从左上角开始，按行变换，还是按列或者 zig-zag，只要所有样本均进行这种处理，它们都是等价的，不会改变向量距离，也即单个点像素距离的累积。

这种二维变一维的转换丢失了很多二维信息，比如水平或垂直方向上像素之间的关系（例如轮廓信息），这与人识别数字的方式是本质不同的，人脑可以把握更本质的图像特征。

