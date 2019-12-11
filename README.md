# 有关连网
- 郑乃千为git设置了一个代理服务器，可以正常使用wget或者是git。
- 如果联网出现问题，随时在微信群中联系即可。

# git 使用的问题
+ 修改代码之前，最好新建一个branch，在branch上commit\push，然后登陆github的网页发pull requests，这样容易merge。
+ 如果修改了代码但是忘了建branch，先stash，切换branch，stash pop，再继续操作。

# 关于 train.py 对 data.py 的调用
+ 我会在data.py中实现一个Dataset的类，训练时需要先调用一次，初始化参数为在本次训练中想使用的数据集的名字的list，例如dataset = Dataset(['maps'])。可以使用的数据集有如下几种：
  + cityscapes：训练集包含2975张图片，用“抽象色块”生成“行车记录仪图像”![](https://github.com/semiwaker/Image-to-Image-Translation-with-Conditional-Adversarial-Network/blob/nkc/etc/cityscapes_sample.jpg)
  + facades：训练集包含400张图片，用“抽象色块”生成“楼的外观"![](https://github.com/semiwaker/Image-to-Image-Translation-with-Conditional-Adversarial-Network/blob/nkc/etc/facades_sample.jpg)
  + maps：训练集包含1096张图片，用“普通地图”生成“卫星地图”![](https://github.com/semiwaker/Image-to-Image-Translation-with-Conditional-Adversarial-Network/blob/nkc/etc/maps_sample.jpg)
  + edges2shoes：训练集包含49825张图片，用“鞋子线稿”生成“鞋子图片”![](https://github.com/semiwaker/Image-to-Image-Translation-with-Conditional-Adversarial-Network/blob/nkc/etc/edges2shoes_sample.jpg)
  + edges2handbags：训练集包含138567张图片，用“手提包线稿”生成“手提包图片”![](https://github.com/semiwaker/Image-to-Image-Translation-with-Conditional-Adversarial-Network/blob/nkc/etc/edges2handbags_sample.jpg)
  + night2day：训练集包含17823张图片，用“白天景色”生成“夜晚景色”（我也不知道为什么数据集叫这个名字）![](https://github.com/semiwaker/Image-to-Image-Translation-with-Conditional-Adversarial-Network/blob/nkc/etc/night2day_sample.jpg)
+ 然后我会把数据加载好，之后你们每次通过调用x_list, y_list = dataset.generate()来获得一组**按数据集随机**的、大小为**BATCH_SIZE**的数据，随机采样与数据格式处理的部分不用train.py来做
+ x_list是一个列表，里面存放着BATCH_SIZE张256\*256\*3的图片，代表最终形态下网络的输入(即文章里的x)
+ y_list是一个列表，里面存放着BATCH_SIZE张256\*256\*3的图片，代表想要生成的图片(即文章中的y)

