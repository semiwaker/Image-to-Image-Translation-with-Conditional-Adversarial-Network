# 抽象模型（可以略过）

生成器接收两个参数作为输入：
	给出的图像x和噪声z
生成器生成一张图G(x,z)

判别器接收两个参数作为输入：
	给出的图像x；真实的相似图像y或生成的图像G(x,z)
判别器目标：
	尽量让 ![](http://latex.codecogs.com/gif.latex?D%28x%2Cy%29%5Cto%201%2C%20D%28x%2CG%28x%2Cz%29%29%5Cto%200)

生成器目标：尽量让 ![](http://latex.codecogs.com/gif.latex?D%28x%2CG%28x%2Cz%29%29%20%5Cto%201)

这是根据损失函数得出的：

![](http://latex.codecogs.com/gif.latex?%5Cbegin%7Baligned%7D%20%5Cmathcal%7BL%7D_%7Bc%20G%20A%20N%7D%28G%2C%20D%29%3D%26%20%5Cmathbb%7BE%7D_%7Bx%2C%20y%7D%5B%5Clog%20D%28x%2C%20y%29%5D&plus;%5C%5C%20%26%20%5Cmathbb%7BE%7D_%7Bx%2C%20z%7D%5B%5Clog%20%281-D%28x%2C%20G%28x%2C%20z%29%29%5D%5Cend%7Baligned%7D)

然后为了让生成器生成的图片类型和我们给出的y的类型尽量相近，增加一个L1 Loss：

![](http://latex.codecogs.com/gif.latex?%5Cmathcal%7BL%7D_%7BL%201%7D%28G%29%3D%5Cmathbb%7BE%7D_%7Bx%2C%20y%2C%20z%7D%5Cleft%5B%5C%7Cy-G%28x%2C%20z%29%5C%7C_%7B1%7D%5Cright%5D)


总目标就是让生成器生成的图片尽量牛逼，目标函数是：

![](http://latex.codecogs.com/gif.latex?G%5E%7B*%7D%3D%5Carg%20%5Cmin%20_%7BG%7D%20%5Cmax%20_%7BD%7D%20%5Cmathcal%7BL%7D_%7Bc%20G%20A%20N%7D%28G%2C%20D%29&plus;%5Clambda%20%5Cmathcal%7BL%7D_%7BL%201%7D%28G%29)


注：实际实现时并没有真的加噪声，而是用随机Dropout当做噪声

# 生成器架构

在原始的Encoder-Decoder的基础上，增加了Skip-connection，就是把Encoder的层的数据整个传输到Decoder的对应层。

可以看下论文的Figure 3。

## 详细架构：

以下有一些论文原话和我的理解

Ck表示有k个卷积核（卷出来有? * ? * k）的Convolution-BatchNorm-ReLU layer
	Convolution就是卷积
	BatchNorm就是批标准化，每一层的每个神经元在拿到输入以后、计算激活函数之前，先把数据归一成均值为0方差为1。
	ReLU就是激活函数
反正这几个都是参数，设置一下就好（？大概
根据下面来看“Ck表示有k个卷积核”可能有点问题，它表示是输出的层数而不是filter个数；加了skip-connection以后有点区别

CDk就是在Ck的基础上增加了Dropout，Dropout率为50%，用来当做噪声。

所有卷积核大小都是4*4，步长为2。



还有一句：“Convolutions in the encoder, and in the discriminator, downsample by a factor of 2, whereas in the decoder they upsample by a factor of 2.”

downsample就是下采样，比如池化Pooling之类的，把一个区域（比如2 * 2）的若干值采样成单个值，factor为2就是让图片的长和宽都除二；（步长为2的卷积本身就可以看成是下采样？）
upsample就是上采样，就是把一个值填充成一个区域，比如把1个值复制4遍变成2 * 2，或者更常见的是把一个2 * 2的反卷积核乘上这个值，然后把这么多个2*2并起来，长宽就都 *2了。



以下是加skip-connection的模型

**encoder:**
**C64-①-C128-②-C256-③-C512-④-C512-⑤-C512-⑥-C512-⑦-C512-⑧**

①之类的表示图像或者卷积核产生的数据，不是卷积核；卷积核filter可以看作是将一层变换到下一层的函数。
变换过来的那个filter有多少个，数据就有几层深。

由于下采样，所以①的长宽是原图的一半，②的长宽是①的一半，③的长宽是②的一半，以此类推。最后产生一个长宽很小（原图的1/256）的、512通道的数据⑧，把它decode。

？：下面我感觉出了点错，因为这边decoder只有7个，下面u-net的decoder有8个。根据⑦和⑨长宽一样所以应该拼起来来看，它应该是在u-net里面多写了一层？

**decoder:**
**⑧-CD512-⑨-CD512-⑩-CD512-⑪-C512-⑫-C256-⑬-C128-⑭-C64-⑮-~C3**

After the last layer in the decoder, a convolution is applied to map to the number of output channels (3 in general,
except in colorization, where it is 2), followed by a Tanh function. 

注：由于是有颜色的图片，所以decoder最后一层输出以后还有个卷积层，（我认为）3个卷积核分别代表rgb（但是这样我就不知道为什么colorization是2个了）
“后面跟着个Tanh”不知道是什么意思，输出0~255之间的值也要激活函数吗qwq

所以最后还有个C3（具体肯定不一样）之类的产生就是最终输出



As an exception to the above notation, Batch-Norm is not applied to the first C64 layer in the encoder.

注：因为它读取的是原图片所以不用Batch-Norm，隐层再用Batch-Norm



All ReLUs in the encoder are leaky, with slope 0.2, while ReLUs in the decoder are not leaky.

注：encoder用的是leaky ReLU, 就是负数部分不是恒0，而也是线性，斜率是 0.2；decoder是普通ReLU



The U-Net architecture is identical except with skip connections between each layer i in the encoder and layer n- i in the decoder, where n is the total number of layers. The skip connections concatenate activations from layer i to layer n - i. This changes the number of channels in the decoder:

（这里我去掉了一个C1024）

**U-Net decoder:**
**CD512(照后面的写法这边应该写1024?)-⑨⑦-CD1024-⑩⑥-CD1024-⑪⑤-C1024-⑫④-C512-⑬③-C256-⑭②-C128-⑮①-~C3**

注：加了skip-connection以后，把encoder中的层拼到decoder里面对应数据层的后，所以Channel数就增加了；但其实filter的个数没有增加，还是和原来的decoder一样的；只是decoder输入的深度增加了（变成了上一层decoder输出再拼起来什么什么）。

# 判别器架构

判别器的神经网络架构就是CNN，但是这里用了PatchGAN的判别方法。不同之处在于：

原来的判别器是输入一张图（或者两张图），输出一个标量代表yes或者no

这里的判别器则是输出一个N*N大小的数组，分别表示某一块（例如70x70的一块）是否是真实的，所以输出是经过sigmoid后的一维数组
那么70 x 70 是怎么来的呢？是通过神经网络架构，通过五层的神经网络以后，最后一层的每一个点的感受野是原图的70x70的块，那么自然就做到了Patchsize=70x70

具体的网络架构如下

C64-C128-C256-C512-C1

其中C128,C256,C512使用BatchNorm；五层的核大小都是4x4，但前面三层的步长为2，后面两层的步长为1，这样反推回去可以算出1<-4<-7<-16<-34<-70，就是70x70的感受野

最后一层C1获得一个29x29x1的数组，把它摊成一维后使用sigmoid激活一下就可以了



