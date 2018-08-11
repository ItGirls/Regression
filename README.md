# Regression
Practice For PRML(Unit 3 Linear Regression)
数据：
data.txt


基本共用：
1）BasisFunction: 
  包含几类常见到基函数，比如高斯、sigmoidal、幂函数

2）measures: 
  包含几个常见评价指标

几个模型及其求解：
1）BayesianRegression: 
  贝叶斯线性回归，引入参数到先验分布。

2）normalEquation: 
  最大似然(用normal Equation实现) & regularized least squares(用normal Equation实现)

3）LeastMeanSquares:
  最小二乘（基于随机梯度下降），不过这个代码目前有问题，还未debug.

主函数：
myRegression： 
    调用这几个模型进行学习（ #1.bayesis Linear regression； #2.normal equation + regularized least squares；#3. least squares with stochastic gradient descent）
