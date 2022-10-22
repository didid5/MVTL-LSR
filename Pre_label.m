function Y = Pre_label(Y,m)
%%---------------------------------------------------------------------------
nClass=length(unique(Y));     % 返回的是和gnd中一样的值，但是没有重复元素。产生的结果向量按升序排序。  对样本标签进行排序
Y_original=zeros(nClass,length(Y));    % 原始的标签矩阵全部为零  

for i=1:m
    if Y(i)==-1
        Y(i)=2;
    end
end

for i=1:m
    for j=1:nClass
        if j==Y(i)
            Y_original(j,i)=1;   % 为有标签的样本赋标签为1
        end  
    end
end
Y=Y_original';   % "'"表示转置