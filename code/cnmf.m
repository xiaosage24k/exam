function [W,H,V,cost_pot]=cnmf(V,options)
err_rat=options.err;
maxiter=options.maxiter;
miniter=options.miniter;
eps=options.eps;
W=options.W0;
H=options.H0;
nl=options.nl;
Label_num=options.Label_num;
numOfDatas=size(V,2);

  %% construct label matrix
   YY=[];
  for ii = unique(Label_num') %reshape(unique(trainLabel),1,C)
    YY = [YY Label_num==ii]; % 为列向量
  end  
  A=zeros(numOfDatas,numOfDatas-nl+options.NL_num); % options.NL_num
  B=ones(1,numOfDatas-nl);
  B=diag(B);
  A(1:nl,1:size(YY,2)) =YY; 
  A(nl+1:end,options.NL_num+1:end)=B;
  A=A';

cost_pot(1,1)=norm(V-W*H*A,'fro')^2; 

%% 开始迭代
for iter=1:maxiter      
    W=W.*(V*A'*H'+eps)./(W*H*A*A'*H'+eps);
    H=H.*(W'*V*A'+eps)./(W'*W*H*A*A'+eps);
%     W=W*(1./diag(sum(W)));
    cost_pot(1,iter+1)=norm(V-W*H*A,'fro')^2;  % 目标函数值
  if iter > miniter   
    if cost_pot(iter+1)-cost_pot(iter) < err_rat  % 当连续两个目标函数的差小于某个值，认为收敛
        break;
    end
  end  
end
 V=H*A; 