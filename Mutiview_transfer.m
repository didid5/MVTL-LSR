function [acc]=Mutiview_transfer(x_1,y1,x_2,y2,C,ratio)

N = size(y2,1);
idx = randperm(N);
n_train = round(N*ratio);

X_test = x_2(idx(1:n_train),:);
X_Train = x_2(idx((n_train+1):end),:);
y_test = y2(idx(1:n_train),:);
y_Train = y2(idx((n_train+1):end),:);

[n1,p1]=size(x_1);
[t1,r1]=size(X_test);
[t2,r2]=size(X_Train);

x1=mat2cell(x_1,[n1],[p1/2 p1/2]);   %  p1/3 p1/3 p1/3    p1/2 p1/2
x_test=mat2cell(X_test,[t1],[r1/2 r1/2]);  %  r1/3 r1/3 r1/3    r1/2 r1/2
x_Train=mat2cell(X_Train,[t2],[r2/2 r2/2]);  %  r2/3 r2/3 r2/3    r2/2 r2/2

%% Compute A' (named as AA) , D' (named as DD) and L for each view
[~,K] = size(x_Train);
AA = cell(1,K);
lamda = 0.01;
for k=1:K
    [s_xtr] = normalizemeanstd( x1{k} );
    L{k} = Construct_L(s_xtr, y1);
    [B,~] = DLE11(y1, s_xtr, lamda);
    AA{k} = B;
end

[max,best_lambda,best_gamma,best_eta] = Max(X_Train,y_Train,K,AA,C);
eta = best_eta;
lambda = best_lambda;
gamma = best_gamma;

[acc] = Accuracy(x_Train,y_Train,x_test,y_test,lambda,gamma,eta,C,AA,K);

end

function [max,best_lambda,best_gamma,best_eta] = Max(X_Train,y_Train,K,AA,C)

gamma_list = 10.^[-3 -2 -1 0 1 2 3];
eta_list=10.^[-1 0 1 2];
lambda_list = 10.^[-2 -1 0 1 2];

m = 1;
max = 0;
ACC = 0;

for lambda_index=1:length(lambda_list)
    lambda=lambda_list(lambda_index);
    
    for gamma_index=1:length(gamma_list)
        gamma=gamma_list(gamma_index);
        for eta_index=1:length(eta_list)
            eta=eta_list(eta_index);
            
            X=[X_Train,y_Train];
            [M,N]=size(X);%数据集为一个M*N的矩阵，其中每一行代表一个样本
            indices=crossvalind('Kfold',X(1:M,N),5);%进行随机分包
            
            for n=1:5%交叉验证k=5，5个包轮流作为测试集
                valid = (indices == n); %获得test集元素在数据集中对应的单元编号
                train = ~valid;%train集元素的编号为非test元素的编号
                X_train=X(train,1:end-1);%从数据集中划分出train样本的数据
                X_valid=X(valid,1:end-1);%test样本集
                y_train=X(train,end);
                y_valid=X(valid,end);
                
                [n2,p2]=size(X_train);
                [n3,p3]=size(X_valid);
                
                x_train=mat2cell(X_train,[n2],[p2/2 p2/2]);  %   p2/3 p2/3 p2/3    p2/2 p2/2
                x_valid=mat2cell(X_valid,[n3],[p3/2 p3/2]);  %   p3/3 p3/3 p3/3    p3/2 p3/2
                
                [Acc,Lambd,Gamm,Et] = Accuracy(x_train,y_train,x_valid,y_valid,lambda,gamma,eta,C,AA,K);
                ACC = Acc + ACC;
            end
            
            ACc(m) =  ACC/5;
            Eta(m) = Et;
            Lambda(m) = Lambd;
            Gamma(m) = Gamm;
            ACC = 0;
            
            if ACc(m) > max
                max = ACc(m);
                best_eta = Eta(m);
                best_lambda = Lambda(m);
                best_gamma = Gamma(m);
            end
            m = m+1;
        end
    end
end
end

function [acc,lambda,gamma,eta] = Accuracy(X_train,y_train,X_valid,y_valid,lambda,gamma,eta,C,AA,K)

n_valid = size(y_valid,1);
A = cell(1,K);
a = size(AA{1},1);
for k=1:K
    A{k} = rand(a, C);
end
W = ones(K,1)*(1/K);
y_2 = Pre_label(y_train,size(y_train,1) );
Niter = 100;
for i=1:Niter
    for k=1:K
        
        [t_xtr,~ ] = normalizemeanstd( X_train{k});
        I=eye(size(t_xtr,2));
        L_T{k} = Construct_L(t_xtr, y_train);
        A{k} = inv(W(k)*(t_xtr'*t_xtr)+lambda*t_xtr'*L_T{k}*t_xtr+gamma*I)*(W(k)*t_xtr'*y_2+gamma*AA{k});
        %compute W
        for j=1:K
            [t_xtr_tmp,~] = normalizemeanstd( X_train{j});
            w(j) = exp( (-1/eta)*norm((t_xtr_tmp*A{j}-y_2),2)^2   );
        end
        W(k) = w(k)/sum(w);
    end
end

%% validation

p=length(y_valid);
for q=1:p
    if y_valid(q)==-1
        y_valid(q)=2;
    end
end

Y_predict  = zeros(n_valid, C);

for k=1:K
    [xte] = normalizemeanstd(X_valid{k});
    Y_predict = Y_predict + W(k)*xte*A{k};
end

con=size(Y_predict,2);
[~,g2]=sort(Y_predict,2);
ord=g2(:,con);
acc=(length(find(ord==y_valid))/length(y_valid));
end