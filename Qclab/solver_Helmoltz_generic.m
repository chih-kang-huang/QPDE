function u = solver_Helmoltz_generic(f, grids, k_val, N_vecs, dx)

dim    = numel(grids);
Nx     = N_vecs(1);

%% 1. Build 1D DFT matrix and correct IDFT
dfmtx = fft(eye(Nx));
GF_1d = dfmtx' / Nx;        

%% 2. Build d-fold DFT/IDFT via iterated Kronecker
FG = dfmtx;
GF = GF_1d;
for k = 2:dim
    FG = kron(FG, dfmtx);
    GF = kron(GF, GF_1d);
end

%% 3. Flatten f row-major (like Python's .flatten())
f_values  = reshape(f, N_vecs);
f_flatten = reshape(permute(f_values, dim:-1:1), [], 1);
f_h       = FG * f_flatten;

%% 4. Build elliptic operator

Elliptic_spec=buildHelmoltzDenom(k_val,N_vecs,dx,dim);
%% 5. Invert and solve
inverse_Elliptic = 1./(Elliptic_spec(:));
inverse_Elliptic=diag(inverse_Elliptic);
u_flatten        = GF * inverse_Elliptic * f_h;

%% 6. Reshape back
u = real(ipermute(reshape(u_flatten, fliplr(N_vecs)), dim:-1:1));

end
