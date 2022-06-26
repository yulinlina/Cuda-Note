### 问题描述
使用cuda实现Mnist的加速 ，包含  
1.数据读入时的矩阵转置  
2.前向计算中的矩阵乘法  
3.累计损失和精确率时的reduction  
4.计算梯度时的矩阵乘法  
5.更新梯度时的矩阵加法  

### 数据
Mnist 数据集  
使用C语言读入  

### 网络
一层神经网络784个神经元，等价于784个权重的线性模型  
激活函数是softmax,损失函数为cross entropy loss

### 梯度计算

### 结果
![结果](https://raw.githubusercontent.com/yulinlina/cudaNote/main/Softmax/gpu%E7%BB%93%E6%9E%9C.png)
