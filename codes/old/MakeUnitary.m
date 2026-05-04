function [M] = MakeUnitary(A)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
N=size(A,1);
M=[A, sqrtm(eye(N)-A'*A); sqrtm(eye(N)-A'*A), -A];
end