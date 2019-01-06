%This is where you can start. 
clear all;
addpath('print');

%% parameters setting
algorithm = 'nmf';       % 'nmf' | 'cnmf'
options.err=1e-6;
options.maxiter=200; % ������
options.miniter=50;  % ��С����
options.eps=1e-9;
r=50;  % ����ά��
rand_num=1;  
folds=2;     % ������֤  
labPer=0.4;  % ��ǩ����

%% load data and normalize it and 
load ORL    % ORL | usps
data=fea'; 
label=gnd;
data=data./sum(data);  % normalize

%% rand_seed and Matrix initialization
rand('state',rand_num); % �����������
[m,n]=size(data);
 Cai=randperm(n);
data=data(:,Cai);  % ��������
label=label(Cai);

%% do NMF
if strcmp(algorithm,'nmf')
     options.W0=rand(m,r);
     options.H0=rand(r,n);  
     [W,V,cost_pot]=nmf(data,options);  % ִ���㷨
elseif strcmp(algorithm,'cnmf') 
     nl=round(n*labPer);
     options.nl=nl;
     options.W0=rand(m,r);
     options.Label_num=label(1:nl);
     options.H0=rand(r,n-nl+numel(unique(options.Label_num)));  
     [W,H,V,cost_pot]  = cnmf(data,options); % ִ���㷨
else
    error('Unsupported algorithm!\n');
end   

%% calculate clustering performance  
if strcmp(algorithm,'nmf')
    c=numel(unique(label)); % �����
    [ac1, nmi1, Pri1,AR1,F1,P1,R1]=printResult(V',label,c,1);  % ����ִ�ж��k-means�����ŵ���ƽ��ֵ�������Ǳ�׼��
elseif strcmp(algorithm,'cnmf') 
    CC=numel(unique(label(nl+1:end,1)));
    [ac1, nmi1, Pri1,AR1,F1,P1,R1]=printResult(V(:,nl+1:end)',label(nl+1:end,1),CC,1);
else
    error('Unsupported algorithm!\n');
end        
     
%% crossValidation for classification performance 
accuracy=crossValidation(V,label,folds,options)
