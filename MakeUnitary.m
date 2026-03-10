function [M] = MakeUnitary(A)
N=size(A,1);
M=[A, sqrtm(eye(N)-A'*A); sqrtm(eye(N)-A'*A), -A];
end