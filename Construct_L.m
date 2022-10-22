function L=Construct_L(X,gnd)
options = [];
% options.NeighborMode = 'KNN';
% options.k = 3;     % 近邻数目
% t=1; 76%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
options.NeighborMode = 'Supervised';
options.gnd = gnd;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
options.WeightMode = 'HeatKernel';
options.t = 10^0;     % 热核参数
W = constructW(X,options);
% [m,n]=size(W);
aa = sum(W);
DDD = diag(aa);    %v=diag(X,k) X为矩阵，v为向量 取矩阵X的第K条对角线元素为向量v
L = DDD-W;
%%
% options = [];
% options.NeighborMode = 'Supervised';
% options.gnd = gnd;
% options.WeightMode = 'HeatKernel';
% options.t = 1;
% W = constructW(fea,options);