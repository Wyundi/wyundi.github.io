---
title: 'Machine Learning: 神经网络'
date: 2019-04-10 10:54:00
tag:
  -机器学习
  -神经网络
---





全连接神经网络基础及相关数学计算。



# Machine Learning: 神经网络



**参考：**

[Machine Learning小结(1)：线性回归、逻辑回归和神经网络](http://blog.kongfy.com/2014/11/machine-learning%E5%B0%8F%E7%BB%931%EF%BC%9A%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%E3%80%81%E9%80%BB%E8%BE%91%E5%9B%9E%E5%BD%92%E5%92%8C%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/)

[神经网络入门_阮一峰](http://www.ruanyifeng.com/blog/2017/07/neural-network.html)

[26种神经网络激活函数可视化_机器之心](https://www.jiqizhixin.com/articles/2017-10-10-3)

[常用激活函数的比较](https://zhuanlan.zhihu.com/p/32610035)

[详解softmax函数以及相关求导过程](https://zhuanlan.zhihu.com/p/25723112)

[代价函数总结](https://blog.csdn.net/juanjuan1314/article/details/78063984)

[CS231n课程笔记翻译：神经网络笔记1（上）](https://zhuanlan.zhihu.com/p/21462488?refer=intelligentunit)

[机器学习（五）- 对于cost function的思考](https://blog.csdn.net/mike112223/article/details/75126402)

[似然与极大似然估计](http://fangs.in/post/thinkstats/likelihood/)

[极大似然估计详解](https://blog.csdn.net/zengxiantao1994/article/details/72787849)

[Softmax函数与交叉熵](<https://zhuanlan.zhihu.com/p/27223959>)

[吴恩达机器学习](https://study.163.com/course/courseMain.htm?courseId=1004570029)

[深度学习工程师_01.神经网络和深度学习](https://mooc.study.163.com/learn/2001281002?tid=2001392029&_trace_c_p_k2_=db6bf4087fc04a09aa4143e4e92e172a)

[卷积神经网络的Python实现](http://www.ituring.com.cn/book/2661)



**环境：**

操作系统：Ubuntu 18.04

神经网络绘图：[NN-SVG](http://alexlenail.me/NN-SVG/index.html)



## **神经网络（Neural Network）**





### **从线性回归和逻辑回归说起**



先给出线性回归和逻辑回归的模型：



线性回归：


$$
h = \theta^Tx
$$


逻辑回归：


$$
z(x) = \theta^Tx
$$

$$
h(x) = \sigma(z)
$$



在这两个模型中都包含$\theta^Tx$。在线性回归中，这个方程表示对实际数据走向的预测。而在逻辑回归中，这个方程表示数据的决策边界。如果数据图像或数据边界近似为一条直线，则可以用一次函数对其进行预测，如果图像近似为一条曲线，就需要用高幂次的方程替换一次函数，才能产生较好的效果。比如，图像近似为一个圆形，就需要用 $h_\theta(x) = g(\theta_0 + \theta_1x1 + \theta_2x2 + \theta_3x_1^2 + \theta_4x_2^2)​$ 这样的圆方程来对其进行预测。我们可以使用非常复杂的模型对数据分布进行预测，或者来适应复杂的决策边界。



但无论是线性回归还是逻辑回归，当特征太多时，计算负荷都会非常大。如果我们有100个特征，并希望用这些特征构建一个非线性的模型，就需要数量非常惊人的特征组合。即使只采用两两组合（$x_1x_2, x_2x_3,...x_{99}x_{100}$），也会有接近5000个特征。这对于一个简单模型来说太多了。这时候我们需要神经网络。





### **神经网络结构**



生物的神经中枢进行思考需要进行以下几个步骤：

> 1. 外部刺激通过神经末梢，转化为电信号，传导到神经细胞（神经元）。
> 2. 无数神经元构成神经中枢。
> 3. 神经中枢综合各种信号，作出判断。

基于对神经元的模拟，建立了称为“感知器”的模型：



![bg2017071202](/images/MachineLearning3_NeuralNetwork/bg2017071202.png)



图中的圆圈即代表神经元（感知器）。它接受多个输入（$x_1, x_2, x_3$），并产生一个输出（$output$）。



多个神经元构成神经网络（神经中枢）：



![bg2017071205](/images/MachineLearning3_NeuralNetwork/bg2017071205.png)



图中每一个圆圈代表一个神经元，所有的神经元相互连接，构成神经网络。这个神经网络接受5个外部输入，数据经过神经网络处理后获得一个输出。





### **逻辑回归和神经网络**



逻辑回归可以看作是一个非常小的神经网络。逻辑回归模型接受输入以后，首先要对数据赋予权重（weight）和偏置（bias），然后使用激活函数（$\alpha(z)$）对其进行激活，并将结果作为输出值。这个过程和神经元的运算过程是一样的，区别只是使用的激活函数不同，所以可以将逻辑回归模型看作是一个只包含一个神经元的神经网络。对于大规模的神经网络，只需要将多个类似逻辑回归的神经元组合在一起即可。



将三个神经元堆叠在一起，然后将其输出作为另一个神经元的输入值，即可产生一个基础的神经网络。



![Screenshot from 2019-03-26 15-27-37](/images/MachineLearning3_NeuralNetwork/Screenshot from 2019-03-26 15-27-37.png)



### **直观理解**



从本质上讲，神经网络能够通过学习得出其自身的一系列特征。在普通的逻辑回归中，我们被限制使用数据中的原始特征$x_1, x_2, x_3, ..., x_n​$，虽然我们可以使用一些二项式来组合这些特征，但仍受这些原始特征的限制，无法表示所有可能的情况。在神经网络中，原始特征只作为输入层，只有第一层神经网络直接利用原始特征作为数据。从第二层开始，所有后面的神经网络的基础数据都是前一层通过学习后得出的新特征，即不受原始特征的限制。所以，更深的神经网络可以学到更多的特征，模拟更复杂的函数。



神经网络中，单个神经元可以表示简单逻辑运算 (and/or/not)。

对于AND运算：



$$
input = x_1, x_2 \in \{0, 1\} \\
output = x_1 \  AND \  x_2
$$


![Screenshot from 2019-03-26 18-02-57](/images/MachineLearning3_NeuralNetwork/Screenshot from 2019-03-26 18-02-57.png)



其中，令$\theta_0 = -3​$，$\theta_1 = 2​$ ，$\theta_2 = 2​$，则$h_{\theta}(x) = sigmoid(-3 + 2x_1 + 2x_2)​$。



则有真值表：

| $x_1$ | $x_2$ | $h_{\theta}(x)$ |
| :---: | :---: | :-------------: |
|   0   |   0   |        0        |
|   0   |   1   |        0        |
|   1   |   0   |        0        |
|   1   |   1   |        1        |



可以看出，$h_{\theta}(x) = x_1 \ AND \ x_2$。



同样：

对于OR运算：取$\theta_0 = -1$，$\theta_1 = 2$ ，$\theta_2 = 2$，则$h_{\theta}(x) = sigmoid(-1 + 2x_1 + 2x_2) = x_1 \ OR \ x_2$；

对于NOT运算：取$\theta_0 = 1$，$\theta_1 = -2$，则$h_{\theta}(x) = sigmoid(1 - 2x_1) = NOT \ x_1​$。



但如果只使用一个神经元，就无法表示复杂的逻辑运算，例如同或 (XNOR)，因为同或不是一个线性可分结构。构造XNOR运算需要使用神经网络。



XNOR运算逻辑如下：


$$
XNOR = (x_1 \ AND \ x_2)OR((NOT \ x_1)AND(NOT \ x_2))
$$


首先构造一个能表达$((NOT \ x_1)AND(NOT \ x_2))$部分的神经元，取$\theta_0 = 3$，$\theta_1 = -2$ ，$\theta_2 = -2$，则$h_{\theta}(x) = sigmoid(3 - 2x_1 - 2x_2) = (NOT \ x_1)AND(NOT \ x_2)$。



然后将该神经元的结果和AND神经元的结果作为输入值构建OR神经元，即可得到能表示XNOR运算的神经网络。





![Screenshot from 2019-03-26 18-55-32](/images/MachineLearning3_NeuralNetwork/Screenshot from 2019-03-26 18-55-32.png)



其中，隐藏层的两个神经元即为$((NOT \ x_1)AND(NOT \ x_2))$部分和AND部分。



通过这种方法，我们就可以构建更复杂的神经网络来模拟更复杂的函数关系。





### **神经网络的表示**



![Screenshot from 2019-03-26 15-27-37](/images/MachineLearning3_NeuralNetwork/Screenshot from 2019-03-26 15-27-37.png)



在上图的神经网络中，共包含（$ x_1, x_2, x_3 $）三个输入变量，他们竖直的堆叠起来成为一层，这是神经网络的输入层。中间四个神经元所产生的结果直接传入下一层的节点中，在训练过程中我们看不到他们产生的结果，所以将他们称为隐藏层。最后一个结点接受隐藏层四个节点的数据，产生最后的结果，这一层称为输出层，它负责产生预测值。通常我们将输入层称为第零层，所以隐藏层为第一层，输出层为第二层。这个神经网络是一个两层的神经网络。



将左侧三个数据（$x_1, x_2, x_3 $）作为输入值，中间四个神经元堆叠在一起成为神经网络的第一层。使用上角标 $X^{[layers] (sample)}$ 分别表示数据在神经网络的第几层/这是第几个样本，下角标 $X_{[features]}$ 表示数据是该层第几个特征。

例如， $x^{[1] (5)}_3$ 表示这个数据是第第一层的第五个样本中第三个特征。



用$a^{[0]}​$来表示输入特征。$a​$表示激活的意思，它意味着网络中本层的值会传递到下一层。输入层将x传递给隐藏层，所以将输入层的激活值称为$a^{[0]}​$。下一层隐藏层同样会产生激活值，将其记为$a^{[1]}​$。按同样的规则，将每层线性方程的结果记为$z​$，参数记为$\theta​$，或$W​$和$b​$。





## **神经网络计算**



### **数据向量化**



为方便计算同时提高运算速度，将所有数据合并为矩阵统一进行运算。对于输入层，将输入的三个特征合并成一个列向量，用$a^{[0]}$表示他们。即：


$$
a^{[0]} = x = 
\left[
\begin{matrix}
x_1 \\
x_2 \\
x_3
\end{matrix}
\right]
=
\left[
\begin{matrix}
a^{[0]}_1 \\
a^{[0]}_2 \\
a^{[0]}_3
\end{matrix}
\right]
$$


$w$的维度取决于从上一层引入的特征数和本层需要产生的特征数。对于隐藏层每一个神经元，都从输入层引入了3个特征，所以每一个神经元都有3个参数 ($w_{n1}, w_{n2}, w_{n3}$)，可以合并成一个 3x1 的矩阵$w_n$。隐藏层共包含4个神经元，就有4个 3x1 的参数 ($w_1, w_2, w_3, w_4$)，将其合并为一个 3x4 大矩阵$W$：


$$
W = 
\left[
\begin{matrix}
w_{11} & w_{21} & w_{31} & w_{41}\\
w_{12} & w_{22} & w_{32} & w_{42}\\
w_{13} & w_{23} & w_{33} & w_{43}
\end{matrix}
\right]
$$


每一个神经元对应一个参数$b$，同样将其合并为矩阵：


$$
b =
\left[
\begin{matrix}
b_1 \\
b_2 \\
b_3 \\
b_4
\end{matrix}
\right]
$$





### **前向传播**



对于每一个神经元，计算过程如下图所示：



![Screenshot from 2019-03-26 15-06-27](/images/MachineLearning3_NeuralNetwork/Screenshot%20from%202019-03-26%2015-06-27.png)



回到两层神经网络，逐个计算每个神经元。



首先使用线性方程计算第一层即隐藏层各节点的$z^{[1]}​$，将结果代入激活函数计算$\alpha^{[1]}​$，并将$\alpha^{[1]}​$作为第二层的输入值传入神经网络的第二层。然后用另一组线性方程和激活函数计算$z^{[2]}​$和$\alpha^{[2]}​$，将$\alpha^{[2]}​$作为输出。这就是一个简单的神经网络前向传播的计算过程。



公式如下：

$$
(a^{[0]}, \theta^{[1]}) \Rightarrow z^{[1]} = W^{[1]T}x + b^{[1]}\Rightarrow \alpha^{[1]} = \sigma(z^{[1]})
$$

$$
(\sigma, \theta^{[2]}) \Rightarrow z^{[2]} = W^{[2]T}\sigma^{[2]} + b^{[2]}\Rightarrow \alpha^{[2]} = \sigma(z^{[2]})
$$



其中，各数据维度如下：

|                Data                 | Dimension |
| :---------------------------------: | :-------: |
|              $a^{[0]}$              |  (3, 1)   |
|              $W^{[1]}$              |  (3, 4)   |
|              $b^{[1]}$              |  (4, 1)   |
| $z^{[1]}, a^{[1]}, \sigma(z^{[1]})$ |  (4, 1)   |
|              $W^{[2]}$              |  (4, 1)   |
|              $b^{[2]}$              |  (1, 1)   |
| $z^{[2]}, a^{[2]}, \sigma(z^{[2]})$ |  (1, 1)   |



以上公式就完成了对一个样本的数据计算，最后给出的结果即为对该样本的预测结果。对于所有样本，同样可以通过向量化简化其运算。



#### **公式整理**



设数据中共有$m​$个样本，每个样本有$n^{[0]}​$个特征。每一个样本的特征可以合并为一个 $(n^{[0]}, 1)​$ 的列向量。将所有样本对应的列向量转置然后堆叠在一起，即得到了一个维度为$ (m, n^{[0]})​$ 的矩阵，X。



建立一个 $l$ 层的神经网络。每一层的特征数（神经元个数）用$n$表示，即为 $(n^{[1]}, n^{[2]}, ..., n^{[l]})​$。



相对应的，公式也应该变为如下形式以和其维度相匹配：


$$
(X, \theta^{[1]}) \Rightarrow z^{[1]} = XW^{[1]} + b^{[1]T}\Rightarrow \alpha^{[1]} = \sigma(z^{[1]})
$$

$$
(\sigma(z^{[1]}, \theta^{[2]}) \Rightarrow z^{[2]} = \sigma^{[2]}W^{[2]} + b^{[2]T}\Rightarrow \alpha^{[2]} = \sigma(z^{[2]})
$$

$$
\vdots
$$

$$
(\sigma(z^{[l-1]}, \theta^{[l]}) \Rightarrow z^{[l]} = \sigma^{[l]}W^{[l]} + b^{[l]T}\Rightarrow \alpha^{[l]} = \sigma(z^{[l]})
$$



数据维度如下：

|                Data                 |       Dimension        |
| :---------------------------------: | :--------------------: |
|                 $X$                 |     $(m, n^{[0]})$     |
|              $W^{[1]}$              |  $(n^{[0]}, n^{[1]})$  |
|              $b^{[1]}$              |     $(n^{[1]}, 1)$     |
| $z^{[1]}, a^{[1]}, \sigma(z^{[1]})$ |     $(m, n^{[1]})$     |
|              $W^{[2]}$              |  $(n^{[1]}, n^{[2]})$  |
|              $b^{[2]}$              |     $(n^{[2]}, 1)$     |
| $z^{[2]}, a^{[2]}, \sigma(z^{[2]})$ |     $(m, n^{[2]})$     |
|              $\vdots$               |        $\vdots$        |
|              $W^{[l]}$              | $(n^{[l-1]}, n^{[l]})$ |
|              $b^{[l]}$              |     $(n^{[l]}, 1)$     |
| $z^{[l]}, a^{[l]}, \sigma(z^{[l]})$ |     $(m, n^{[l]})$     |





### **激活函数**



在神经网络中，激活函数决定来自给定输入集的节点的输出，其中非线性激活函数允许网络模拟复杂的非线性行为。同时，激活函数需要是（几乎完全）可微分的，才可以通过梯度下降对神经网络进行优化。此外，复杂的激活函数可能会产生梯度消失或梯度爆炸等问题。



使用一个神经网络时，需要决定使用哪种激活函数用在隐藏层上，哪种用在输出节点上。前面用到的sigmoid函数在某些情况下效果很好，但在某些情况下会出问题，这时可以使用其他激活函数。



常见的激活函数包括Sigmoid、ReLU、Linear等，还有一些特殊的激活函数如Softmax。



#### **Sigmoid**

**公式：**


$$
\alpha = \sigma(z) = \frac{1}{1 + e^{-z}}
$$


**图像：**



![sigmoid](/images/MachineLearning3_NeuralNetwork/sigmoid.png)



Sigmoid函数可以将实际的输出值“挤压“到0到1的范围内，适合输出为概率的情况。但由于其存在一些问题，现在已经很少有人在构建神经网络的过程中使用Sigmoid函数了。



**存在的问题：**



1. Sigmoid函数饱和使梯度消失。当Sigmoid值接近0和1时，其导数会接近0，那么求得的梯度也会接近0，这会导致算法在反向传播的过程中没有信号传回上一层，即梯度消失。
2. Sigmoid函数的输出不是零中心的。Sigmoid函数产生的结果总是正数且平均值为0.5而不是0，这会使神经网络收敛的速度更慢。因为下一层神经网络的输入值不是零中心，就更容易产生饱和，导致梯度消失。





#### **Tanh**

**公式：**


$$
\alpha = 2\sigma(2z) - 1 = \frac{1 - e^{-z}}{1 + e^{-z}}
$$


**图像：**



![tanh](/images/MachineLearning3_NeuralNetwork/tanh.png)



Tanh是将Sigmoid函数的范围扩展到了 (-1, 1)，这就解决了SIgmoid函数输出不是零中心的问题。但仍然存在函数饱和的问题。结果表明，如果在隐藏层上使用tanh，结果总是优于sigmoid函数。但是在输出层上，如果需要处理二分类问题，就需要y的值处于0和1之间而不是-1和1之间。此时需要使用sigmoid函数。



为了防止饱和，现在主流的做法会在激活函数前多做一步batch normalization，尽可能保证每一层网络的输入值具有均值较小，零中心的分布。



#### **ReLU**

**公式：**




$$
\alpha = max(0, z)
$$


**图像：**

![ReLU](/images/MachineLearning3_NeuralNetwork/ReLU.png)



修正线性单元（Rectified linear unit，ReLU）函数是神经网络中最常用的激活函数。ReLU函数模仿了生物神经元的特性，即只有在输入超过阈值时才激活神经元。并且在输入为正时，导数不为0，从而允许基于梯度下降的优化。因为无论是其函数还是导数都不包含复杂的数学运算，所以使用ReLU函数也极大的减少了计算时间。



然而，当输入值为负时，ReLU函数的梯度为0，从而其权重无法得到更新，并且在此后的训练过程中该神经元会一直保持沉默，即神经元死亡。但是，有足够的隐藏层使ReLU函数的输入值大于0，所以对于大多数训练数据来说学习过程仍然可以很快。



如果在隐藏层上不确定使用哪个激活函数，那么通常会使用ReLU函数。



还有另一个版本的ReLU函数被称为Leaky ReLU。当z是负值时，其函数值不是0，而是轻微的倾斜。



**图像：**



![LeakyReLU](/images/MachineLearning3_NeuralNetwork/LeakyReLU.png)



由于其在负半区梯度不为0，所以不会产生ReLU函数在负半区神经元死亡的问题。这个函数通常比ReLU函数效果要好，但并不常用。



#### **Softmax**

**公式：**


$$
S_i = \frac{e^{z_i}}{\sum\limits_{j = 1}^n e^{z_j}}
$$


**示意图如下：**



![softmax](/images/MachineLearning3_NeuralNetwork/softmax.jpg)



Softmax函数是Logistic回归模型在多分类问题上的推广，适用于多分类且不同类别之间互斥的问题。当类别数 k = 2 时，Softmax函数退化为Logistic回归。Softmax函数可以将多个神经元的输出映射到 (0, 1) 区间，可以看作当前输出是其属于各分类的概率，从而解决多分类的问题。



对Softmax函数求导：



当 $i = j$ 时：


$$
\frac{\partial S_i}{\partial z_j} = \frac{\partial \frac{e^{z_i}}{\sum\limits_{k = 1}^{n} e^{z_k}}}{\partial z_j} = \frac{e^{z_i}\sum\limits_{k = 1}^{n} e^{z_k} - e^{z_i}e^{z_j}}{[\sum\limits_{k = 1}^{n} e^{z_k}]^2} = \frac{e^{z_i}}{\sum\limits_{k = 1}^{n} e^{z_k}}\frac{\sum\limits_{k = 1}^{n} e^{z_k} - e^{z_j}}{\sum\limits_{k = 1}^{n} e^{z_k}} = S_i(1 - S_j)
$$


当 $i \neq j​$ 时：


$$
\frac{\partial S_i}{\partial z_j} = \frac{\partial \frac{e^{z_i}}{\sum\limits_{k = 1}^{n} e^{z_k}}}{\partial z_j} = \frac{0 - e^{z_i}e^{z_j}}{[\sum\limits_{k = 1}^{n} e^{z_k}]^2} = -\frac{e^{z_i}}{\sum\limits_{k = 1}^{n} e^{z_k}}\frac{e^{z_j}}{\sum\limits_{k = 1}^{n} e^{z_k}} = -S_iS_j
$$



即：


$$
\frac{\partial S_i}{\partial z_j} = 
\begin{cases}
S_i(1 - S_j) & \text{if}\ i = j \\
-S_iS_j & \text{if}\ i \neq j 
\end{cases}
$$



结果可用雅可比矩阵表示：



$$
\frac{\partial S_i}{\partial z_j} = 
 \left[
 \begin{matrix}
   \frac{\partial S_1}{\partial z_1} & \frac{\partial S_1}{\partial z_2} & \cdots &     	\frac{\partial S_1}{\partial z_n}\\
   \frac{\partial S_2}{\partial z_1} & \frac{\partial S_2}{\partial z_2} & \cdots &     	\frac{\partial S_2}{\partial z_n}\\
   \vdots & \vdots & \ddots & \vdots \\
   \frac{\partial S_n}{\partial z_1} & \frac{\partial S_n}{\partial z_2} & \cdots &     	\frac{\partial S_n}{\partial z_n}\\
  \end{matrix}
  \right]
  =
   \left[
 \begin{matrix}
   S_1(1 - S_1) & -S_1S_2 & \cdots & -S_1S_n\\
   -S_2S_1 & S_2(1 - S_2) & \cdots & -S_2S_n\\
   \vdots & \vdots & \ddots & \vdots \\
   -S_nS_1 & -S_nS_2 & \cdots &     	S_n(1 - S_n)\\
  \end{matrix}
  \right]
$$





### **代价函数**



代价函数是神经网络模型优化时的目标，通过最小化代价函数来优化模型。对于不同的任务类型，神经网络模型需要使用不同的代价函数。



常见的有代价函数有均方差代价函数、对数损失函数、交叉熵等。



#### **均方差代价函数**

公式：


$$
J_{\theta} = \frac{1}{2m}\sum\limits_{i = 1}^{m}[h_{\theta}(x^{(i)}) - y^{(i)}]^2
$$


式中，$h_{\theta}(x^{(i)})$表示对第$i$个样本的预测值，$y^{(i)}$表示该样本对应的真实值。$\frac{1}{2}$的作用是简化后面的求导运算。



均方差代价函数适用于回归模型。当预测值越接近真实值，代价函数会越接近0。



计算梯度如下：


$$
\frac{\partial}{\partial\theta_n}J_{\theta} = \sum\limits_{i = 1}^{m}[(h_{\theta}(x^{(i)}) - y^{(i)})x^{(i)}_n]
$$


式中，当 $n$ 取值为 (1, n) 时，$\theta_n$对应为$w_n$，$x^{(i)}_n$对应该样本的第 $n$ 个特征。当 $n$ 取值为0时，$\theta_n$即 $\theta_0$ 对应 $b$，取$x^{(i)}_0 = 1$。



#### **对数损失函数**

公式：


$$
Loss(y, P(y | x)) = -logP(y | x)
$$


其中：


$$
P(y | x) = 
\begin{cases}
h_\theta(x) & \text{if}\ y = 1 \\
1 - h_\theta(x) & \text{if}\ y = 0 
\end{cases}
$$



所以，$Loss(y, P(y|x))$ 可简化为：



$$
Loss(y, P(y | x)) = -y * log(h_\theta(x)) - (1 - y) * log(1 - h_\theta(x))
$$



即取 y 和 (1 - y) 表示两种情况，y = 1 时，(1 - y) 为0，y = 0 时，(1 - y) 为1。



对应的代价函数为：


$$
J(\theta) = -\frac{1}{m}\sum\limits_{i = 1}^{m}Loss = -\frac{1}{m}\sum\limits_{i = 1}^{m}[y^{(i)} * log(h_\theta(x^{(i)})) + (1 - y^{(i)}) * log(1 - h_\theta(x^{(i)}))]
$$



当 y = 1且 h(x) = 1 时误差为零，且当 h(x) < 1 时Loss随h的减小而增大；当 y = 0 且 h(x) = 0 时误差为零，且当 h(x) > 1 时Loss随h增大而增大。



对数损失函数适用于二分类问题，即逻辑回归。其本身是由统计学中最大似然估计得出。在其激活函数使用Sigmoid函数的情况下，计算梯度如下：


$$
\frac{\partial}{\partial\theta_n}J_{\theta} = \frac{1}{m}\sum\limits_{i = 1}^{m}[(h_{\theta}(x^{(i)}) - y^{(i)})x^{(i)}_n]
$$



此式和均方差代价函数梯度表达式相同，但其中的$h_{\theta}(x^{(i)})$不同，所以两个代价函数不同。



#### **交叉熵**

概率:


$$
P(x | \theta)
$$


其含义为在已知 $\theta$ 的条件下，最后的结果为 $x$ 的可能性。



最大似然估计：


$$
L(\theta | x)
$$


其含义为在已知结果为 $x$ 的条件下，取环境变量为 $\theta$ 时结果为 $x$ 的可能性。即：


$$
L(\theta | x) = P(x | \theta)
$$



对于多个可能的结果，最大似然估计的表达式如下：


$$
L(\theta) = P(x_1, x_2, ..., x_m | \theta) = p(x_1 | \theta)p(x_2 | \theta)...p(x_m | \theta) = \prod\limits_{i = 1}^{m}P(x_i | \theta)
$$



根据定义可知，概率和最大似然估计的区别为：$L$ 是关于 $\theta$ 的函数，$P$ 是关于 $x$ 的函数。在神经网络中，我们需要根据已知事件来找出产生这种结果的最有可能的条件，目的是根据这个最有可能的条件去推测未知事件。最大似然函数在现有数据的条件下，通过调整环境变量 $\theta$ 来获取最大可能性，最后产生的结果即可用于对未知数据的预测。



为便于计算，用 $log$ 函数将最大似然估计中的累乘转化为求和：


$$
log(L(\theta)) = log(\prod\limits_{i = 1}^{m}P(x_i | \theta)) = \sum\limits_{i = 1}^{m}log(P(x_i | \theta))
$$



观察 $log$ 函数图像可知，当 $L(\theta) = 1$ 时，$log(L(\theta)) = 0$；当 $L(\theta) \rightarrow 0 $ 时，$log(L(\theta)) \rightarrow -\infty$。



我们需要优化 $\theta$ 使得 $L(\theta)$ 尽可能接近1，所以定义损失函数如下：


$$
Loss(\theta) = -log(L(\theta)) = -\sum\limits_{i = 1}^{m}log(P(x^{(i)} | \theta))
$$


以上即为交叉熵。



为了让结果更趋向于实际值，取：


$$
P(x^{(i)} | \theta) = 
\begin{cases}
h_\theta(x_j) & \text{if}\ y_k = 1 \\
0 & \text{if}\ y_k = 0 
\end{cases}
$$


则：


$$
Loss(\theta) = -\sum\limits_{i = 1}^{m}log(P(x^{(i)} | \theta)) = -\sum\limits_{i = 1}^{m}\sum\limits_{k = 1}^{n}[y_klog(h_{\theta}(x_k^{(i)})]
$$



可以看出，逻辑回归中的对数损失函数是交叉熵取 $n = 2$ 即只有两个分类时的特例。



使用交叉熵作为多分类神经网络的代价函数：


$$
J(\theta) = \frac{1}{m}Loss(\theta) = -\frac{1}{m}\sum\limits_{i = 1}^{m}\sum\limits_{k = 1}^{n}[y_klog(h_{\theta}(x_k^{(i)})]
$$







### **反向传播**



反向传播过程也和逻辑回归过程很相似，需要通过代价函数和梯度下降来优化每一层的参数。不同的是，由于神经网络每一层相对独立，且层数较多，不能直接用代价函数对所有参数进行求导。所以需要使用高等数学中的链式法则，对每一层分别求导。



使用在前向传播过程中构建的神经网络：



> 设数据中共有$m​$个样本，每个样本有$n^{[0]}​$个特征。建立一个 $l​$ 层的神经网络。每一层的特征数（神经元个数）用$n​$表示，即为 $(n^{[1]}, n^{[2]}, ..., n^{[l]})​$。



前向传播过程如下：


$$
(X, \theta^{[1]}) \Rightarrow z^{[1]} = XW^{[1]} + b^{[1]T}\Rightarrow \alpha^{[1]} = \sigma(z^{[1]})
$$

$$
(\sigma(z^{[1]}, \theta^{[2]}) \Rightarrow z^{[2]} = \sigma^{[2]}W^{[2]} + b^{[2]T}\Rightarrow \alpha^{[2]} = \sigma(z^{[2]})
$$

$$
\vdots
$$

$$
(\sigma(z^{[l-1]}, \theta^{[l]}) \Rightarrow z^{[l]} = \sigma^{[l]}W^{[l]} + b^{[l]T}\Rightarrow \alpha^{[l]} = \sigma(z^{[l]})
$$



以 $X$ 作为输入数据，$\sigma(z^{[l]})$即为最后的预测值。



使用代价函数计算预测值和真实值之间的差距。



[...未完待续]