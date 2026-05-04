clear; clc; close all;

%% 1. Simulation Configuration
dim = 2;           
n  = 6;
N=2^n;
k=10
A=eye(2);

f_handle = @(x,y) cos(2*pi*x) .* sin(-4*pi*y);
u_true=1;

fprintf('Starting %dD Simulation with N=%d...\n', dim, N);

[grids, N_dir, dx] = GenericHelmoltz_QPDE(f_handle, k, N, dim, u_true)

fprintf('Done.\n');