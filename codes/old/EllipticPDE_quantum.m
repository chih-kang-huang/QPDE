%% Spectral method setup

% Domain boundaries
x_lb = 0;
x_rb = 1;

% Grid size: N = 2^n points per dimension (FFT-friendly)
n = 6;
N = 2^n;
d=2;
L = 1; 
dx = L / N;
% Elliptic operator coefficient matrix (constant, d = 2)
% A = [3 1;
%      1 2];
A=eye(d);
% Domain length and grid spacing


% Periodic 1D grid (endpoint excluded, as in spectral methods)
x = x_lb + (0:N-1) * dx;

% 2D computational grid with "ij" indexing
% ndgrid is the MATLAB equivalent of meshgrid(..., indexing='ij')
[xs, ys] = ndgrid(x, x);


FG=GroupFourier(d,n);
GF=FG.ctranspose();

% Build derivative operator in spectral domain
D = diag(spectral_eigenvalues(N));

% Identity matrix
I = eye(N);

% Elliptic operator in spectral space
Elliptic_spec = ...
    A(1,1) * kron(D^2, I) + ...
    A(1,2) * kron(D, D) + ...
    A(2,1) * kron(D, D) + ...
    A(2,2) * kron(I, D^2);

% Avoid division by zero (zero-frequency mode)
Elliptic_spec(1,1) = 1;

invElliptic=diag(1./diag(Elliptic_spec));

DiagEncoding=MakeUnitary(invElliptic);

totalCircuit=qclab.QCircuit(d*n+1);
totalCircuit.push_back(GF);
totalCircuit.push_back(qclab.qgates.MatrixGate(0:2*n,DiagEncoding,"Diagonal"))
totalCircuit.push_back(FG);

totalMat=totalCircuit.matrix;


totalCircuit.draw


%% Classical RHS
% f = @(x,y) cos(2*pi*x) .* sin(-2*pi*y);
f = @(x,y) cos(2*pi*x) .* sin(-4*pi*y);

u = solver_Elliptic(f, xs, ys, A, N, N, dx);
f_flatten = f(xs, ys);
f_flatten = f_flatten(:);
% u_anal=f(xs,ys).*(-1/(20*(pi^2)));
res=totalMat(1:2^(2*n),1:2^(2*n))*f_flatten;

reshapedres=reshape(res,N,N);
norm(u-reshapedres)/norm(u)
% norm(reshapedres-u_anal)/norm(u_anal)