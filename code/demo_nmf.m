%This is where you can start. 
clear all;

%% parameters setting
options.err=1e-6;
options.maxiter=200; % 最大迭代
options.miniter=20;  % 最小迭代
options.eps=1e-9;
r=100;
rand_num=1;  

%% load data and normalize it and 
load ORL   
V=fea'; 
label=gnd;
V=V./sum(V);  % normalize

%% rand_seed and Matrix initialization
rand('state',rand_num); % 设置随机种子
[m,n]=size(V);
W0=rand(m,r);
H0=rand(r,n);
V=V(:,randperm(n));
label=label(randperm(n));

%% do NMF
[W,H,cost_pot]=nmf(V,W0,H0,options);

%% plot and calculate accuracy 
plot(cost_pot);
train_num=round(n/2);
results = knnclassify(H(:,train_num+1:end)', H(:,1:train_num)', label(1:train_num)', 1);
accuracy = mean(results == label(train_num+1:end))