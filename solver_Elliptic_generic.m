function u = solver_Elliptic_generic(f, grids, A, N, dx)
%   Spectral solver for -div(A grad u) = f
%   with periodic boundary conditions on [0,1]^d.
%
%   Inputs:
%     f     - RHS as a flat column vector (N^d x 1)
%     grids - cell array of spatial grids from ndgrid (used only for dim)
%     A     - d x d diffusion coefficient matrix
%     N     - grid size vector [Nx, Ny, ...]
%     dx    - grid spacing (scalar, uniform in all directions)
%
%   Output:
%     u     - solution on the N-D grid (same shape as f reshaped)

dim    = numel(grids);
N_vecs = N;

f_values = reshape(f, N_vecs);
f_h      = fftn(f_values);

denom = buildEllipticDenom(A, N_vecs, dx, dim);

u_h = f_h ./ denom;
u   = real(ifftn(u_h));

end