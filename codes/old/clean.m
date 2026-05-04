function M= clean(M)
eps=10e-10;
[n,m]=size(M);
for i=1:n
    for j=1:m

        if abs(M(i,j))<eps
            M(i,j)=0;
        end
    end
end

end