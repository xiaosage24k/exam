function [W,H,cost_pot]=nmf(V,W0,H0,options)
err_rat=options.err;
maxiter=options.maxiter;
miniter=options.miniter;
eps=options.eps;
W=W0;
H=H0;
cost_pot(1,1)=norm(V-W*H,'fro')^2;
for iter=1:maxiter      % 开始迭代
    H=H.*(W'*V+eps)./(W'*W*H+eps);
    W=W.*(V*H'+eps)./(W*H*H'+eps);
%     W=W*(1./diag(sum(W)));
    cost_pot(1,iter+1)=norm(V-W*H,'fro')^2;  % 目标函数值
  if iter > miniter   
    if cost_pot(iter+1)-cost_pot(iter) < err_rat  % 当连续两个目标函数的差小于某个值，认为收敛
        break;
    end
  end  
end