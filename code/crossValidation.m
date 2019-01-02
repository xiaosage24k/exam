function [accuracy,clustering,W,H,cost_pot] = crossValidation( data, label, folds, options)

%% allocation
[m,numOfDatas] = size(data);  % ������
clustering=[];
accuracy=0;  
     
%% prepare croos validation
crossindex = crossvalind('kfold',numOfDatas,folds);   % ���ɽ�����֤����,ָ��= crossvalind��'Kfold'��N��K������������ɵ�ָ��������N�ι۲��K��������֤��

for i=1:folds         % 5�ν�����֤
    %% divide data into training set and testing set
    partitionIndex = find(crossindex==i);    % �ҵ�λ�ã�crossindexΪnumOfDatas��С��ȡֵΪ1:5��ֵ���൱�ڷ���5������
    testLabel = label(partitionIndex);  % ÿ��ѡ��һ��fold�Ӽ���Ϊ����
    trainDataArray = data;     % ȡ��ѵ������
    trainLabel = label;
    trainLabel(partitionIndex) = [];  % ѵ���Ѹò���ɾ��
    
    %�ֱ�ȡ�ò��Ժ�ѵ��������
        testDataArray = data(:,partitionIndex);
        trainDataArray(:,partitionIndex) = [];

    %% calculate classification performance
     results = knnclassify(trainDataArray, trainDataArray, trainLabel, 1);
     accuracy = accuracy + mean(results == testLabel);
end
accuracy = accuracy / folds;  % ����ȡƽ����ֻ�о��ȵ�ָ��
end
