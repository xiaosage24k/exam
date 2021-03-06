# 2018-2019第一学期期末考核

#### 考核说明：给定cls文件[（点这里）](https://github.com/xiaosage24k/exam/tree/master/tex)，答题内容写在latex上，双栏，其他包、格式自己填写，图片的要求和周报告一样。
### 1.文献整理、公式编写题：
 * [1.1](http://www.4243.net/)  下载引用CNMF的最新的2篇会议论文和2篇期刊论文，修改下载文件的名字（如下图），截图附在PDF上。
 ![image](https://github.com/xiaosage24k/exam/blob/master/images/%E4%B8%8B%E8%BD%BD%E6%96%87%E7%8C%AE%E5%91%BD%E5%90%8D%E6%A0%BC%E5%BC%8F.jpg)
 
 * 1.2 写成参考文献的格式。
 * 1.3 用latex编写出NMF和CNMF的目标函数的公式。注意： 矩阵大写黑正体，向量小写黑正体，变量用斜体。 
### 2.matlab实践题：给定NMF、CNMF基本代码、交叉验证函数、分类聚类指标测试函数[（点这里）](https://github.com/xiaosage24k/exam/tree/master/code)和两个数据集[（点这里）](https://github.com/xiaosage24k/exam/tree/master/dataset)：
* 2.1 分别测试NMF和CNMF（50%标签）在两个数据集上的3fold交叉验证的分类和聚类结果，把结果绘制成表格（如下图）。 注意：分类只有1个指标，而聚类有7个指标；分类和聚类做成两个表；对于CNMF的聚类，只测试训练时没有用到标签的部分。
![image](https://github.com/xiaosage24k/exam/blob/master/images/%E8%A1%A8%E6%A0%BC1.jpg)
### 3.结果展示题：
* 3.1 把2.1中NMFCNMF的分类精度绘制成下图一样的柱状图。横坐标为数据集，纵坐标为精度。
![image](https://github.com/xiaosage24k/exam/blob/master/images/%E6%9F%B1%E7%8A%B6%E5%9B%BE.png)
* 3.2 在ORL数据集上分别测试包含10%、20%、30%标签的CNMF的聚类精度，并绘制出类似下面的曲线图。 横坐标为标签比例，纵坐标为聚类精度。     
        ![image](https://github.com/xiaosage24k/exam/blob/master/images/%E6%9B%B2%E7%BA%BF%E5%9B%BE.jpg)
