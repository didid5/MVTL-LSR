function [A,M,obj]=DLE11(gnd,X,lambda)
%% X(nXm);Y(nXc);A(mXc);||XA-Y-B.*M||+lambda*Tr(A'*X'*L*XA)
%%
num_l=length(gnd);
% nue_1是标签长度
Y = Pre_label(gnd,num_l);  
% 构建标签矩阵Y
B=Construct_B(Y);
% 构建奢华矩阵B
L=Construct_L(X,gnd);  
Niter=400;
[x33,y33]=size(Y);
M=rand(x33,y33);
tempp=inv(X'*X+lambda*X'*L*X+0.01*eye(size(X,2)));
for iter=1:Niter
%     iter
    A=tempp*(X'*(Y+B.*M));
    F=X*A-Y;
    M_original=B.*F;
    [x11,y11]=size(F);
    %%
    for i=1:x11
    for j=1:y11
        M(i,j)=max(M_original(i,j),0);
    end
    end  
   
%     M=M_original;
    %%
    sam1=X*A-Y-(B.*M);
    sam2=A'*X'*L*X*A;
    obj(iter)=trace(sam1'*sam1)+lambda*trace(sam2);
    if iter>2
        if abs(obj(iter)-obj(iter-1))<0.01
            break
        end
    end
end




