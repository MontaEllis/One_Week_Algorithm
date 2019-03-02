#### 2 逻辑回归算法梳理
[2019.3.1 ~ 2019.3.3]
##### 1、逻辑回归与线性回归的联系与区别
逻辑回归有一个Sigmoid函数，能将样本映射到(0,1)上，便于分类
线性回归是预测，逻辑回归是分类

##### 2、 逻辑回归的原理
面对一个回归或者分类问题，建立代价函数，然后通过优化方法迭代求解出最优的模型参数，然后测试验证我们这个求解的模型的好坏

##### 3、逻辑回归损失函数推导及优化
**逻辑回归**：
$h(x)  = \frac{1} {1 + e^{-(w^Tx+b)}}$
设$y_i$=1的概率为$p_i$，$y_i$=0的概率为$1-p_i$
则观测概率：
$p(y_i) = p_i^{y_i}(1-p_i)^{1-y_i}$
概率由逻辑回归公式求解，带进去得到极大似然函数：
$\prod_i^N h(x_i)^{y_i} * (1-h(x_i))^{1-y_i}$
取**对数**：
$L(w) = \sum_i( y_i * logh(x_i) + (1-y_i) * log(1 - h(x_i)) )$
$= \sum y_i(logh(x_i)-log(1-h(x_i))) + log(1-h(x_i))$
$= \sum y_ilog\frac{h(x_i)}{1-h(x_i)} + log(1-h(x_i))$
$= \sum y_i(w^Tx_i) + log(1 - \frac{1}{1+e^{-wx_i}})$
$= \sum_i( y_i * (w^Tx_i) - log(1 + e^{w^Tx_i}) )$

梯度下降，牛顿法
##### 4、 正则化与模型评估指标
在原来的损失函数基础上加上权重参数的平方和等来惩罚
平均均方误差MSE
拟合优度Goodness of fit
##### 5、逻辑回归的优缺点
优点：
容易实现，速度快，适合于二分类
缺点：对于非线性等问题表现不明显

##### 6、样本不均衡问题解决办法
将二分类转化为其他问题
赋予正负例不同权重系数

##### 7. sklearn参数
**Sklearn.linear_model.**
penalty:’l1’ or ‘l2’ ,默认’l2’ #惩罚

dual:bool 默认False ‘双配方仅用于利用liblinear解算器的l2惩罚。’

tol: float, 默认: 1e-4 ‘公差停止标准’

C:float 默认:1.0 正则化强度， 与支持向量机一样，较小的值指定更强的正则化。

fit_intercept: bool 默认:True 指定是否应将常量（a.k.a. bias或intercept）添加到决策函数中。

intercept_scaling:float ,默认:1 仅在使用求解器“liblinear”且self.fit_intercept设置为True时有用。 在这种情况下，x变为[x，self.intercept_scaling]，即具有等于intercept_scaling的常数值的“合成”特征被附加到实例矢量。 截距变为intercept_scaling * synthetic_feature_weight

class_weight: dict or ‘balanced’ 默认:None

 与{class_label：weight}形式的类相关联的权重。 如果没有给出，所有类都应该有一个权重。“平衡”模式使用y的值自动调整与输入数据中的类频率成反比的权重，如n_samples /（n_classes * np.bincount（y））。请注意，如果指定了sample_weight，这些权重将与sample_weight（通过fit方法传递）相乘。

random_state:int,RandomState实例或None，可选，默认值：None

在随机数据混洗时使用的伪随机数生成器的种子。 如果是int，则random_state是随机数生成器使用的种子; 如果是RandomState实例，则random_state是随机数生成器; 如果为None，则随机数生成器是np.random使用的RandomState实例。 在求解器=='sag'或'liblinear'时使用。