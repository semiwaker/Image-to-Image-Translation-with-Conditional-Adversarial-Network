# 有关连网
- 郑乃千为git设置了一个代理服务器，可以正常使用wget或者是git。
- 如果联网出现问题，随时在微信群中联系即可。

# git 使用的问题
+ 修改代码之前，最好新建一个branch，在branch上commit\push，然后登陆github的网页发pull requests，这样容易merge。
+ 如果修改了代码但是忘了建branch，先stash，切换branch，stash pop，再继续操作。

# 关于 train.py 对 data.py 的调用
+ 我会在data.py中实现一个Dataset的类，训练时需要先调用一次，初始化参数为在本次训练中想使用的数据集的名字的list(这个我会之后写明)，例如dataset = Dataset(['maps'])
+ 然后我会把数据加载好，之后你们每次通过调用data = dataset.generate()来获得一组按数据集随机的、大小为BATCH_SIZE的数据，随机采样与数据格式处理的部分不用train.py来做
+ 关于数据最终格式我们可以再讨论，目前我使用原project的方式，直接把input和label拼在一起形成512\*256的图片，然后转成numpy数组返给你们。