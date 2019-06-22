## Startdt AI提出：使用生成对抗网络用于One-Stage目标检测的知识蒸馏方法

原创： 洪伟 [CVer](javascript:void(0);) *今天*

点击上方“**CVer**”，选择加"星标"或“置顶”

重磅干货，第一时间送达![img](https://mmbiz.qpic.cn/mmbiz_jpg/ow6przZuPIENb0m5iawutIf90N2Ub3dcPuP2KXHJvaR1Fv2FnicTuOy3KcHuIEJbd9lUyOibeXqW8tEhoJGL98qOw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![img](https://mmbiz.qpic.cn/mmbiz_png/yNnalkXE7oVLvFpn3zdTUgMOFNdfD5Ybq3l511F4x3B7G4JA4exXxrE0CzqXfByDP5ziaarViaJpLUjhhlciaGtDg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



Date：20190621

作者团队：**Startdt AI Lab& 智云视图**

论文链接：**https://arxiv.org/abs/1906.08467**

github：https://github.com/p517332051/GAN-Knowledge-Distillation-SSD



## **摘要**

卷积神经网络对目标检测的精度有着显著的提升，并且随着卷积神经网络的深度加深，对目标检测精度提升也越大，但是也需要更多的浮点计算。许多研究者通过知识蒸馏的方法，通过把一个更深更大的教师网络中的知识转移到一个小的学生网络中，以提高学生网络在目标检测中的效果。而大部分知识蒸馏的方法都需要设计复杂的代价函数，并且多数针对两步目标检测算法，本文针对一步目标检测算法提出一个干净有效的知识蒸馏方案。将教师网络生成的特征层作为真实样本，学生网络生成的特征层做为假样本，并对两者做生成对抗训练，以提高学生网络在一步目标检测中的表现。

**1 Introduction**

近些年来，随着目标检测算法的发展，研究者们发现利用更深更大的卷积神经网络作为骨架，对目标检测算法的精度提升越大。并且随着目标检测算法的检测精度提升，使视觉检测算法逐渐从非关键性领域，走向关键性领域（比如无人驾驶和医疗等领域）。但是为了保证检测精度，不得不使用更大的卷积神经网络作为骨架，造成检测速度下降，计算设备成本增加。因此许多研究者在确保检测精度的前提下，提高检测速度提出了很多方法和总结，如通过深度分离卷积 [1，2],或者通过点群卷积(pointwise group convolution)和通道混洗(channel shuffle)[3, 4] 来降低卷积神经网络浮点运算次数的方法，在保证骨架网络精度和容量的情况下减少计算量。虽然获得可观的提速效果，但是这些方法需要精心设计和调整骨架网络。很多研究者认为更深的骨架网络虽然有着更大的网络容量，因此在图像分类、目标检测等任务上有着更优秀的表现。但是一些特定的任务并不需要这么大的容量，所以在保证卷积神经网络精度的情况和下，对卷积神经网络做压缩、量化、通道减枝等[5, 6, 7, 8, 9]。

另一方面，有关于知识蒸馏的工作表明[10, 11, 12, 13]，使用一个更深更大的模型，并且在充分训练完毕后作为teacher net，然后再选取一个比较浅的模型作为student net，最后使用teacher net输出的结果或者中间结果作为soft label结合真实样本的true label同时训练student net，可以极大的提升student net在特定任务上的表现。但是大部分这些方法都需要设计非常复杂的代价函数和训练方式，并且这些方法多用于图像分类和两步目标检测等，极少用于一步目标检测。因此，我们需要一个更加简单而有效，并且可以适用于一步目标检测的知识蒸馏方式。本文提出一种简单而有效知识蒸馏神经网络架构，并且可以明显的提升student net在一步目标检测网络的表现。和常规的知识蒸馏方式不同的是，我们参考对抗生成的神经网络架构[14]，将重型目标检测神经网络和轻型目标检测神经网络的骨架分别拆分出来作为teacher net和student net，然后把teacher net 生成的feature map作为真实样本，而student net则作为生成器，并把student net生成的feature map作为假样本，最后根据真实样本和假样本设计一个神经网络作为判别器，做生成对抗训练。

**我们的贡献主要有两点：**

1 提出一种不需要设计复杂的代价函数的网络架构，并且可以适用于一步目标检测。

2 利用对抗生成网络架构，避免复杂的知识迁移设计，让student net自动的从teacher net中获取暗知识，

3 我们的实验表明，该方法可以直观有效的提高student net在一步目标检测中的表现。

**2 Related Works**

深度学习目标检测算法架构主要分为两种，一种是一步检测，比如Liu W等人提出的SSD[15]，直接通过通过卷积神经网络回归出物体的位置和类别，另一种是二步检测，如girshick等人提出的fast rcnn[16]，以及后来Faster-RCNN [17] and R-FCN [18]等，首先通过卷积神经网络回归候选框，最后根据候选框再次识别每个候选框的类别，并回归出正确的位置。

网络裁剪，许多研究者认为深度神经网络被过度参数化，并且有很多冗余的神经元和连接，He Y等人认为[8],cnn每层神经元都是稀疏的，利用lasso regression回归找出cnn每层最有代表性的神经元重构该层的输出。Zhuang Z等人[9]认为layer-by-layer进行通道剪枝会影响cnn的鉴别能力，所以通过在fine-tune和剪枝阶段加入辅助loss，来保留cnn每层的鉴别能力。

网络量化, Wu J等人[20]通过k-means聚类算法加速和压缩模型的卷积层和全连接层，通过减小每层输出响应的估计误差可实现更好的量化结果，并提出一种有效的训练方案抑制量化后的多层累积误差 。Jacob B[21]等人提出将weights和inputs量化为uint8 bias量化为unit32同时训练期间前向时候采用量化，反向修正误差不量化，以确保cnn表现的情况下提高inference速度。

知识蒸馏 是一种压缩模型并确保准确的一种方法。hinton 等人提出[2]将teacher net输出的结果作为soft label，并提倡使用温度交叉熵而不是L2损失。romero 等人[19]认为需要更多的unlabeled data让student net去mimic才能使student net经可能的接近teacher net，Chen G[12]等人在优化2步目标检测网络分别将teacher net的中间feature map 以及rpn/rcnn的暗知识提取出来让student net去mimic。其他研究者也有将teacher net的attention信息给student网络，如Zagoruyko S[22]等人提出spatial-attention，将teacher net的热力信息传递给student net。Yim J等人[23]将teacher net层与层之间的关系作为student网络mimic的目标。但是他们设计的的知识蒸馏都是要设计非常复杂的loss function，和复杂的暗知识的提取方式，并且这些方法多是在两步目标检测算法中很少用于一步目标检测中。为了能用一个简单有效的知识蒸馏的方式，我们参考生成对抗网络的架构方式[14]将教师网络生成的特征层作为真实样本，学生网络生成的特征层做为假样本，并对两者做生成对抗训练，以提高学生网络在一步目标检测中的表现。

![img](https://mmbiz.qpic.cn/mmbiz_png/yNnalkXE7oVLvFpn3zdTUgMOFNdfD5Yb6hVnP2V9gmZgN62kPBH1kgBcvO6Z8bMW80NNgczlPVERgArFoOcCkw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**3 Method**

在本文中，我们采用一步目标检测算法SSD[15]作为我们的目标检测算法,SSD目标检测算法结构主要分成两部分，1）骨架网络,作为特征提取器。2）Head,在骨架网络提取的特征上，检测出目标的类别和位置。为了能获取更好的知识蒸馏效果，合理利用这个两个部分至关重要。

**3.1 Overall Structure**

fig 1为我们算法模型的整体结构，我们首先使用一个容量更大的SSD模型，在充分训练后将该SSD模型拆分成骨架网络和SSD-Head，其中骨架网络作为teacher net，然后再挑选一个容量较小的CNN作为student net。我们把teacher net生成的多个feature map作为true sample，而student net生成的多个feature map作为fake sample，并且将true sample和fake sample送入D Net中相对应的每个判别网络(fig 2)中，同时把fake sample输入到SSD-Head中。

![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

**3.2 Training Process**

(1)

![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)公式1中的N代表batchsize的大小，D代表判别网络，Teacher和Student分别代表teacher net和student net, θt、θs、θd分别代表teacher net、student net 和D Net模块中每个判别网络的weights。Lconf表示SSD中分类的损失函数，Lloc表示SSD中边界框的损失函数。

**4 Experiment**

在本章节，我们将在PASCAL VOC中做实验来验证我们的方法，包含20个类别。并且我们的方法训练的硬件为two NVIDIA GTX 1080Ti GPUs。训练所用的软件框架为gluoncv。



**4.1 Training and testing data**

由于时间的关系，我们训练使用的数据集Pascal Voc 2012trainval和Pascal Voc 2007 trainval sets，测试数据集为Pascal Voc 2007 test sets。该数据集包含检测物体的类别和位置信息。评估标准按照Pascal Voc竞赛所约定的，通过iou=0.5时的mAP来评估模型检测精度。而coco数据集上，使用coco 2017 trainset作为训练集，coco 2017 test作为测试集。

**4.2 Results**

我们将原生的SSD和在不同的Teacher net下知识蒸馏的SSD做比较，最高可以提升student net 2.8mAP。不过有趣的是，当teacher net为ResNet101，student net为ResNet18时，提升的效果反而不如ResNet50。而在coco上使用resnet50作为teacher net，moblinet作为student net，提升Moblient-SSD 4个mAP。

![img](data:image/gif;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQImWNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg==)

Table 1. Different student nets are not used GAN-knowledge distillation and the use of a GAN-knowledge distillation in different teacher net test results.

![img](https://mmbiz.qpic.cn/mmbiz_png/yNnalkXE7oVLvFpn3zdTUgMOFNdfD5Yb0RAQ45DKgO9rH5qJIsZ3IpfRbGKGjFDOko1ic8GoNrDU5GiaAOib76FdA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

Table 2. moblienetv1 use GAN-knowledge distillation in coco.

**2019年6月19日**

目前已经将该方法使用在faster rcnn上,考虑到时间，目前仅仅在pascal voc 2007上进行测试，coco正在训练。

![img](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oVLvFpn3zdTUgMOFNdfD5Ybun6cL3NzVxBOtAmAvs4xxUfPQqQBtlyDYcC3kYbhpqTRzBzSm7BjLA/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

Table 3. Teacher net为骨架网络为ResNet101的faster rcnn，且使用Pascal Voc 2007 trainval作为训练集，在Pascal Voc 2007 test测试集上mAP为74.8+。第一行和第二行使用GAN Knowledge Distillation[1]方法，第三行为cvpr2019的 *Distilling Object Detectors with Fine-grained Feature Imitation*[2]的方法效果。

**这是在SSD上训练的log**

https://github.com/p517332051/GAN-Knowledge-Distillation-SSD/blob/master/ssd_512_mobilenet1_0_resnet50_v1_voc_g_d_new_train.log

**这是在faster_rcnn上的训练log**

https://github.com/p517332051/GAN-Knowledge-Distillation-SSD/blob/master/faster_rcnn_resnet50_resnet101_v1b_g_d_voc_train.log

日后会在下面网址放出模型和代码，欢迎大家 star/fork

**https://github.com/p517332051/GAN-Knowledge-Distillation-SSD**

**CVer-目标检测交流群**



扫码添加CVer助手，可申请加入**CVer-目标检测群。****一定要备注：****研究方向+地点+学校/公司+昵称**（如目标检测+上海+上交+卡卡）

![img](https://mmbiz.qpic.cn/mmbiz_jpg/yNnalkXE7oX7pdpBKibicSnmb8wRGicbT0Rhr61k0f922lbXcowibk5DTRibROvFB1yMCAZQvj1iaEe6Qsia9bU0UMJCA/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

▲长按加群



这么硬的**论文分享**，麻烦给我一个在**在看**



![img](https://mmbiz.qpic.cn/mmbiz_png/e1jmIzRpwWg3jTWCAZ4BrnvIuN20lLkhIjtg4GRSDhTk9NpeF0GGTJwUpKPatscIQU7Ndj9hgl8BPpGj2BJoFw/640?tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

▲长按关注我们

**麻烦给我一个在看****！**

[阅读原文](https://mp.weixin.qq.com/s?__biz=MzUxNjcxMjQxNg==&mid=2247489934&idx=2&sn=38dfa89a80ac5704e7e88a34a4469af5&chksm=f9a26b01ced5e21760ee059a15d0abf1fba2a146c4b108fb310d1ee79f84c08bf74e92e39b15&mpshare=1&scene=1&srcid=&key=90581f21d61583cc4e169b6fd87dc85174fb1ae03a881aef396dff311de61a68437ee695b6753bdb65c79712c3ce6d17a53a832456ac7ec53533d32188284732722db1ec600e725207bc4b935a01549f&ascene=1&uin=MjMzNDA2ODYyNQ%3D%3D&devicetype=Windows+10&version=62060833&lang=zh_CN&pass_ticket=9IRx4xaiTOM91hbVb0vdLdZS7OAeVm9843tvTCrnf4TH0GaPL9OCge%2B%2FWfjP4WI%2F##)





微信扫一扫
关注该公众号