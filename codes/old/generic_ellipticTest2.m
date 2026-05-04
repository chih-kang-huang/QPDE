%% Spectral method setup

% Domain boundaries
x_lb = 0;
x_rb = 1;

% Grid size: N = 2^n points per dimension (FFT-friendly)
n =5;
N = 2^n;
d=2;
% Elliptic operator coefficient matrix (constant, d = 2)
% A = [3 1;
%      1 2];
A=eye(d);
% Domain length and grid spacing
L  = x_rb - x_lb;
dx = L / N;

% Periodic 1D grid (endpoint excluded, as in spectral methods)
x = x_lb + (0:N-1) * dx;

% 2D computational grid with "ij" indexing
% ndgrid is the MATLAB equivalent of meshgrid(..., indexing='ij')
[xs, ys] = ndgrid(x, x);

totalMat= QPDE_Generator(A,n);




%% Classical RHS
% f = @(x,y) cos(2*pi*x) .* sin(-2*pi*y);
f = @(x,y) cos(2*pi*x) .* sin(-4*pi*y);
f1= @(x,y,z) cos(2*pi*x) .* sin(-4*pi*y).*cos(2*pi*z);
u = solver_Elliptic(f, xs, ys, A, N, N, dx);

f_flatten = f(xs, ys);
f_flatten = f_flatten(:);

[xs, ys,zs] = ndgrid(x, x,x);
grids = struct('x', xs, 'y', ys,'z',zs);
N_dir = struct('x', N, 'y', N,'z',N);
A=eye(3);
u_generic = solver_Elliptic_generic(f1, grids, A, N_dir, dx);


% u_anal=f(xs,ys).*(-1/(20*(pi^2)));

size(totalMat)
2^(2*n)
res=totalMat(1:2^(2*n),1:2^(2*n))*f_flatten;

reshapedres=reshape(res,N,N);
norm(u-reshapedres)/norm(u);

% norm(u-u_generic)
% norm(reshapedres-u_anal)/norm(u_anal)
figure

%% LEFT: visualize u (slices)
subplot(1,2,1)
imagesc(u'); 
axis square; colorbar;
title('Classical Solution');
xlabel('x'); ylabel('y'); set(gca, 'YDir', 'normal');

%% RIGHT: isosurface of u_generic
subplot(1,2,2)
hold on

nx = size(u_generic,1);
ny = size(u_generic,2);
nz = size(u_generic,3);

slice(u_generic,[],[],1:nz)
slice(u_generic,1:nx,[],[])
slice(u_generic,[],1:ny,[])
shading interp

axis equal tight
colorbar
title('Solution u (slices)')
xlabel('x'); ylabel('y'); zlabel('z')
view(3)

% A = u_generic;
% minA = min(A(:));
% maxA = max(A(:));
% iso  = (minA + maxA)/2;
% 
% p = patch(isosurface(A, iso));
% isonormals(A, p)
% 
% set(p, ...
%     'FaceColor', 'red', ...
%     'EdgeColor', 'none', ...
%     'FaceAlpha', 0.7)
% 
% daspect([1 1 1])
% view(3)
% axis tight
% camlight headlight
% lighting gouraud
% title('Isosurface of u\_generic')
