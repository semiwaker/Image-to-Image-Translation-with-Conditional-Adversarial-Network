# Image to Image Translation with Conditional Adversarial Network 图像翻译条件对抗网络

这个项目是以Tensorlayer实现的[Image to Image Translation with Conditional Adversarial Network ](https://arxiv.org/abs/1611.07004).

## 模型

模型分为两部分：生成器（Generator）和判别器（Discriminator）。

其中，生成器的任务是生成看上去尽量真实的图像，而判别器的任务是将生成器生成的图像和真实的图像区分开。

生成器接收两个参数作为输入：给出的图像 x 及 噪声 z，
	并生成一张新的图像 G(x,z) 作为输出。

判别器接收两个参数作为输入：给出的图像 x 及 真实的相似图像 y 或 生成器输出的图像 G(x,z)
	输出一个介于0与1之间的标量 D(x,y) 或 D(x, G(x, z)) 表示判别器认为的真实程度。值越大则认为越真实。

所以：
	判别器的目标是尽量让 ![](http://latex.codecogs.com/gif.latex?D%28x%2Cy%29%5Cto%201%2C%20D%28x%2CG%28x%2Cz%29%29%5Cto%200)
	生成器的目标是尽量让 ![](http://latex.codecogs.com/gif.latex?D%28x%2CG%28x%2Cz%29%29%20%5Cto%201)

以这个目的而设置的损失函数：

![](http://latex.codecogs.com/gif.latex?%5Cbegin%7Baligned%7D%20%5Cmathcal%7BL%7D_%7Bc%20G%20A%20N%7D%28G%2C%20D%29%3D%26%20%5Cmathbb%7BE%7D_%7Bx%2C%20y%7D%5B%5Clog%20D%28x%2C%20y%29%5D&plus;%5C%5C%20%26%20%5Cmathbb%7BE%7D_%7Bx%2C%20z%7D%5B%5Clog%20%281-D%28x%2C%20G%28x%2C%20z%29%29%5D%5Cend%7Baligned%7D)

为了让生成器生成的图片类型和我们给出的 y 的类型尽量相近，增加一个L1 Loss：

![](http://latex.codecogs.com/gif.latex?%5Cmathcal%7BL%7D_%7BL%201%7D%28G%29%3D%5Cmathbb%7BE%7D_%7Bx%2C%20y%2C%20z%7D%5Cleft%5B%5C%7Cy-G%28x%2C%20z%29%5C%7C_%7B1%7D%5Cright%5D)

而总的目标就是让生成器生成的图片尽量真实，目标函数是：

![](http://latex.codecogs.com/gif.latex?G%5E%7B*%7D%3D%5Carg%20%5Cmin%20_%7BG%7D%20%5Cmax%20_%7BD%7D%20%5Cmathcal%7BL%7D_%7Bc%20G%20A%20N%7D%28G%2C%20D%29&plus;%5Clambda%20%5Cmathcal%7BL%7D_%7BL%201%7D%28G%29)

注：实际实现时并没有真的传入噪声 z，而是用随机 Dropout 当做噪声

### 生成器架构

在原始的Encoder-Decoder的基础上，增加了Skip-connection，就是把Encoder的层的数据整个传输到Decoder的对应层。

***这里需要有能人士贴一个论文Figure3的图QAQ***

### 判别器架构

判别器是使用了PatchGAN的CNN。它与普通判别器的不同在于：

原来的判别器是输入两张图像，输出一个标量表示可信程度；

PatchGAN判别器则是输出一个正方形的张量，张量中的每个数据表示图像中某一块（例如70x70的一块）的可信程度。
为了达到这个目的，CNN层被小心地设计，以使得输出张量中的每个点的感受野是设置好的值（例如70x70)。

## 参数

### 一些约定：

Ck表示有k个卷积核的Convolution-BatchNorm-ReLU layer。

CDk就是在Ck的基础上增加了Dropout，Dropout率为50%，用来当做噪声。

所有卷积核大小都是4*4，步长为2。

在Encoder以及判别器中，每一层CNN以2的速度下采样（每一层边长变为1/2）；

在Decoder中，每一层CNN以2的速度上采样。

### 生成器具体参数

默认图像输入是 256 x 256。

**encoder:**
**C64--C128-C256-C512-C512-C512-C512-C512-**

输入通过encoder之后变为1x1x512的张量。

下面是未加U-Net的decoder：

**decoder:**
**-CD512-CD512-CD512-C512-C256-C128-C64-C3**

最后的C3是为了变为RGB图像。

所有Encoder中的ReLU都是Leaky ReLU，斜率是0.2；decoder中的ReLU是不Leaky的ReLU。

**U-Net decoder:**
**CD1024-CD1024-CD1024-C1024-C512-C256-C128-C3**

注：加了skip-connection以后，把encoder中的层拼到decoder里面对应数据层的后，所以Channel数就增加了；但其实filter的个数没有增加，还是和原来的decoder一样的；只是decoder输入的深度增加了。

### 判别器具体参数

**Discriminator:**
**C64-C128-C256-C512-C1**

其中C128,C256,C512使用BatchNorm；

五层的核大小都是4x4，但前面三层的步长为2，后面两层的步长为1。
这样反推回去可以算出1<-4<-7<-16<-34<-70，就是70x70的感受野。

最后一层C1之后得到一个29x29x1的张量，把它摊成一维后使用sigmoid激活即可。

## 要求

``` sh
python 3.7.3
numpy 1.16.0
pandas 0.25.3
matplotlib 3.1.2
tensorflow-gpu 2.0.0
tensorlayer 2.1.0
opencv-python 4.1.2.30
```

## 使用方法

### 准备数据

TODO: fill this section

如果你只想使用我们已经训练好的权重，进行检验

```sh
cd src
./install.sh
```

install.sh脚本将自动创建model文件夹，并下载解压权重。

### 训练

``` sh
python train.py dataset [-v] [-g] [-e EPOCH] [--gpu=GPU]
```

其中，dataset为数据集的名字，-v选项为详细输出，-g选项为绘制loss折线图，-e选项为指定Epoch数，--gpu选项为指定使用的GPU。

训练好的权重会以hdf5格式存在`.\model\G_model_dataset.hdf5`和`.\model\D_model_dataset.hdf5`里面，其中dataset是输入的数据集名字。第一次训练之前需要先`mkdir model`。

### 查看结果

``` sh
python evaluate.py dataset [-t train/test] [-i index1 [index2 ...]] [--gpu=GPU]
```

其中，dataset是数据集的名字，-t选项可以指定查看训练集还是测试集（默认为测试集），-i选项可以指定查看的图片编号列表（默认为全部），--gpu选项可以指定使用的GPU。

evaluate.py会将指定的数据集中指定的图片取出，使用训练好的生成器获得对应的输出，将输入、标签和输出排列好后以图片格式储存到以数据集命名的文件夹里面。需要提前`mkdir`。

## 结果

我们在facades数据集、edges2shoes数据集和edges2handbags数据集上取得了较好的成果。

facades数据集的输入是建筑的表面布局，输出是建筑物的正面图片。

![facades1](images/facades1.jpg)

![facades2](images/facades2.jpg)

其中大部分能输出较好的建筑正面，但是也出现了一些问题：

![facades3](images/facades3.jpg)

数据集中有不少图片是有缺陷的，我们的分类器不够鲁棒，没法很好地理解标签图中的缺陷。数据增强能减轻图片缺陷的问题，但是不能消除这个问题。

edge2handbags数据集的输入是各种包的线图，输出是实物的照片。

![handbag1](images/handbag1.jpg)

![handbag2](images/handbag2.jpg)

鲁棒性依然是问题，如下图中背景不够干净时，会出现多余的东西。

![handbag3](images/handbag3.jpg)

以及领域相关的问题：学不出包上的人脸图像，凡是出现人脸时效果都不好。

![handbag4](images/handbag4.jpg)

在night2day数据集上，我们的效果比较不好。night2day数据集的输入是白天的图片，输出是同一位置夜晚的图片。由于建筑物较暗，而天空较亮，基本只学到了天空的部分。

![night2day](images/night2day.jpg)
