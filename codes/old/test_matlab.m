n=3;
N=2^n;
res=zeros(N,N);
f =@(x,y) cos(2*pi*x)*sin(-2*pi*y)


for i=1:N
    for j=1:N
    res(i,j)=f((i-1)/N,(j-1)/N);
    end
end
res

reshaped=reshape(res',[],1);
reshaped(1:4);
dft=kron(dftmtx(N),dftmtx(N));
dft(:,[2,4,10,12])
tmp=dft*reshaped;
%figure(1)
heatmap(real(reshape(tmp,N,N)))
%figure(2)
%%heatmap(imag(reshape(tmp,N,N)))

subplot(1,2,1)
heatmap(real(reshape(tmp,N,N)))
subplot(1,2,2)
heatmap(imag(reshape(tmp,N,N)))


