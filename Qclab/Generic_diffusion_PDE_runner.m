%% Main Script to Test GenericQuantumDiffusion
clear; clc; close all;

%% 1. Simulation Configuration
dim = 2;           
n  = 6;
N=2^n;
dt  = 1e-3;        
steps = 100;       

if dim == 2
    A = eye(2);
    A=[3,1;1,2];
elseif dim == 3

    A = eye(3); 
end


if dim == 2
% Source Term: f(x,y) = cos(2*pi*x) * sin(-4*pi*y)
    f_handle = @(x,y) cos(2*pi*x) .* sin(-4*pi*y);
    
    % Initial Condition: u0(x,y) = cos(2*pi*x)sin(8*pi*y) + 2sin(6*pi*y) + 3sin(10*pi*x)cos^2(12*pi*y)
    u_handle = @(x,y) cos(2*pi*x) .* sin(8*pi*y) + ...
                      2 * sin(6*pi*y) + ...
                      3 * sin(10*pi*x) .* (cos(12*pi*y).^2);
    % u_true=@(x,y) cos(2*pi*x) .* sin(-4*pi*y);

elseif dim == 3

    f_handle = @(x,y,z) cos(2*pi*x).*sin(-4*pi*y).*cos(2*pi*z)

    u_handle = @(x,y,z) cos(2*pi*x).*sin(8*pi*y)+2*sin(6*pi*y)+3*sin(10*pi*x).*cos(12*pi*y).^2
    % u_true=@(x,y,z) cos(2*pi*x) .* sin(-4*pi*y);
end


fprintf('Starting %dD Simulation with N=%d...\n', dim, N);


GenericDiffusion_QPDE(f_handle, u_handle, A, N, dim, dt, steps);%,u_true

fprintf('Done.\n');