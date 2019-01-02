function [accuracy,clustering,W,H,cost_pot] = crossValidation( data, label, folds, options)

%% allocation
[m,numOfDatas] = size(data);  % 样本数
clustering=[];
accuracy=0;  
     
%% prepare croos validation
crossindex = crossvalind('kfold',numOfDatas,folds);   % 生成交叉验证索引,指数= crossvalind（'Kfold'，N，K）返回随机生成的指数，用于N次观测的K倍交叉验证。

for i=1:folds         % 5次交叉验证
    %% divide data into training set and testing set
    partitionIndex = find(crossindex==i);    % 找到位置，crossindex为numOfDatas大小的取值为1:5的值，相当于分了5个分区
    testLabel = label(partitionIndex);  % 每次选择一个fold子集作为测试
    trainDataArray = data;     % 取得训练数据
    trainLabel = label;
    trainLabel(partitionIndex) = [];  % 训练把该部分删除
    
    %分别取得测试和训练的样本
        testDataArray = data(:,partitionIndex);
        trainDataArray(:,partitionIndex) = [];

    %% calculate classification performance
     results = knnclassify(trainDataArray, trainDataArray, trainLabel, 1);
     accuracy = accuracy + mean(results == testLabel);
end
accuracy = accuracy / folds;  % 精度取平均，只有精度的指标
end
