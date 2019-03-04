#### 3 决策树算法梳理
[2019.3.3 ~ 2019.3.5]
##### 1. 信息论基础（熵 联合熵 条件熵 信息增益 基尼不纯度） 
熵：描述一个时间的不确定性
联合熵：AB同时发生的信息熵
条件熵：A发生时B的信息熵
信息增益：决策树算法中是用来选择特征的指标
基尼不纯度：将来自集合中的某种结果随机应用于集合中某一数据项的预期误差率

##### 2.决策树的不同分类算法（ID3算法、C4.5、CART分类树）的原理及应用场景 
* ID3算法
ID3决策树可以有多个分支，但是不能处理特征值为连续的情况。决策树是一种贪心算法，每次选取的分割数据的特征都是当前的最佳选择，并不关心是否达到最优
* C4.5
ID3的基础上改进而提出的
* CART分类树
改进了前两种算法中的一个缺点：使用信息增益或信息增益比时，可选值多的特征往往有更高的信息增益

##### 3. 回归树原理 
回归是为了处理预测值是连续分布的情景，其返回值应该是一个具体预测值。回归树的叶子是一个个具体的值，从预测值连续这个意义上严格来说，回归树不能称之为“回归算法”。因为回归树返回的是“一团”数据的均值，而不是具体的、连续的预测值（即训练数据的标签值虽然是连续的，但回归树的预测值却只能是离散的）。所以回归树其实也可以算为“分类”算法，其适用场景要具备“物以类聚”的特点，即特征值的组合会使标签属于某一个“群落”，群落之间会有相对鲜明的“鸿沟”。利用回归树可以将复杂的训练数据划分成一个个相对简单的群落，群落上可以再利用别的机器学习模型再学习。
##### 4. 决策树防止过拟合手段 
剪枝
##### 5. 模型评估 
![f58c2320cfce66ef24fb7ad6e37e2d92.png](en-resource://database/792:0)
##### 6. sklearn参数详解，Python绘制决策树
class sklearn.tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2,min_samples_leaf =1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None,class_weight=None, presort=False)


* criterion:string类型，可选（默认为"gini"）
    衡量分类的质量。支持的标准有"gini"代表的是Gini impurity(不纯度)与"entropy"代表的是information gain（信息增益）。
* splitter:string类型，可选（默认为"best"）
    一种用来在节点中选择分类的策略。支持的策略有"best"，选择最好的分类，"random"选择最好的随机分类。
* max_features:int,float,string or None 可选（默认为None）
    在进行分类时需要考虑的特征数。
    1.如果是int，在每次分类是都要考虑max_features个特征。
    2.如果是float,那么max_features是一个百分率并且分类时需要考虑的特征数是int(max_features*n_features,其中n_features是训练完成时发特征数)。
    3.如果是auto,max_features=sqrt(n_features)
    4.如果是sqrt,max_features=sqrt(n_features)
    5.如果是log2,max_features=log2(n_features)
    6.如果是None，max_features=n_features
    注意：至少找到一个样本点有效的被分类时，搜索分类才会停止。
* max_depth:int or None,可选（默认为"None"）
    表示树的最大深度。如果是"None",则节点会一直扩展直到所有的叶子都是纯的或者所有的叶子节点都包含少于min_samples_split个样本点。忽视max_leaf_nodes是不是为None。
 * min_samples_split:int,float,可选（默认为2）
    区分一个内部节点需要的最少的样本数。    
    1.如果是int，将其最为最小的样本数。
    2.如果是float，min_samples_split是一个百分率并且ceil(min_samples_split*n_samples)是每个分类需要的样本数。ceil是取大于或等于指定表达式的最小整数。
* min_samples_leaf:int,float,可选（默认为1）
    一个叶节点所需要的最小样本数：
    1.如果是int，则其为最小样本数
    2.如果是float，则它是一个百分率并且ceil(min_samples_leaf*n_samples)是每个节点所需的样本数。
* min_weight_fraction_leaf:float,可选（默认为0）
    一个叶节点的输入样本所需要的最小的加权分数。
* max_leaf_nodes:int,None 可选（默认为None）
    在最优方法中使用max_leaf_nodes构建一个树。最好的节点是在杂质相对减少。如果是None则对叶节点的数目没有限制。如果不是None则不考虑max_depth.
* class_weight:dict,list of dicts,"Banlanced" or None,可选（默认为None）
    表示在表{class_label:weight}中的类的关联权值。如果没有指定，所有类的权值都为1。对于多输出问题，一列字典的顺序可以与一列y的次序相同。
    "balanced"模型使用y的值去自动适应权值，并且是以输入数据中类的频率的反比例。如：n_samples/(n_classes*np.bincount(y))。
    对于多输出，每列y的权值都会想乘。
    如果sample_weight已经指定了，这些权值将于samples以合适的方法相乘。
* random_state:int,RandomState instance or None
    如果是int,random_state 是随机数字发生器的种子；如果是RandomState，random_state是随机数字发生器，如果是None，随机数字发生器是np.random使用的RandomState instance.
* persort:bool,可选（默认为False）
    是否预分类数据以加速训练时最好分类的查找。在有大数据集的决策树中，如果设为true可能会减慢训练的过程。当使用一个小数据集或者一个深度受限的决策树中，可以减速训练的过程。