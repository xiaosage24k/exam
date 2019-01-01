function [ accuracy] = crossValidation( data, label, folds, options)

%% allocation
numOfTypes = size(data,2);     % �ӽ���
numOfDatas = size(data{1},2);  % ������
accuracy = 0; 

%% prepare croos validation
crossindex = crossvalind('kfold',numOfDatas,folds);   % ���ɽ�����֤����,ָ��= crossvalind��'Kfold'��N��K������������ɵ�ָ��������N�ι۲��K��������֤��

for i=1:folds         % 5�ν�����֤
    %% divide data into training set and testing set
    partitionIndex = find(crossindex==i);    % �ҵ�λ�ã�crossindexΪnumOfDatas��С��ȡֵΪ1:5��ֵ���൱�ڷ���5������
    testDataArray = cell(1,numOfTypes);    % ����������Ϊ�ռ�
    testLabel = label(partitionIndex);  % ÿ��ѡ��һ��fold�Ӽ���Ϊ����
    trainDataArray = data;     % ȡ��ѵ������
    trainLabel = label;
    trainLabel(partitionIndex) = [];  % ѵ���Ѹò���ɾ��
    label_all=[trainLabel testLabel];  % FOR PSLF
    nl=size(trainLabel,2);
    
    for view = 1:numOfTypes   %�ֱ�ȡ�ò��Ժ�ѵ��������
        testDataArray{view} = data{view}(:,partitionIndex);
        trainDataArray{view}(:,partitionIndex) = [];
        trainData_pslf{view}=[trainDataArray{view} testDataArray{view}]; %���첿�ֱ�ǩ�����ݼ�����2foldʱΪ50%
    end
    %% do NMF

   
    %% calculate accuracy 

end

accuracy = accuracy / folds;  % ����ȡƽ����ֻ�о��ȵ�ָ��
end
