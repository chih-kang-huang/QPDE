%% Main Script to Test GenericQuantumDiffusion
clear; clc; close all;


dim = 3;            
n  = 5;
N=2^n;


if dim == 2
    %A = eye(2);
    A=[100000,0;0,1];
elseif dim == 3
    A = eye(3); 
end



if dim == 2

    f_handle = @(x,y) cos(2*pi*x) .* sin(-4*pi*y);
    u_true=@(x,y) -cos(2*pi*x).*sin(-4*pi*y)/(20*pi^2);
 
elseif dim == 3

    f_handle = @(x,y,z) 5 * sin(2*pi*x) .* sin(2*pi*y) .* sin(2*pi*z);
    u_true=@(x,y,z) -cos(2*pi*x).*sin(-4*pi*y)/(20*pi^2);
 
end

fprintf('Starting %dD Simulation with N=%d...\n', dim, N);

GenericElliptic_QPDE(f_handle, A, N, dim,u_true)%

fprintf('Done.\n');