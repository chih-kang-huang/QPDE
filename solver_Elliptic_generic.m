function u = solver_Elliptic_generic(f, grids, A, N_vecs, dx)

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

%% 4. Build 1D eigenvalue diagonal matrix
L    = Nx * dx;
k    = spectral_eigenvalues(Nx, true, L);
D    = diag(k);
Imat = eye(Nx);

%% 5. Build elliptic operator
N_total       = Nx^dim;
Elliptic_spec = zeros(N_total);
for i = 1:dim
    for j = 1:dim
        if A(i,j) ~= 0
            mats = repmat({Imat}, 1, dim);
            mats{i} = mats{i} * D;
            mats{j} = mats{j} * D;
            term = mats{1};
            for kk = 2:dim
                term = kron(term, mats{kk});
            end
            Elliptic_spec = Elliptic_spec + A(i,j) * term;
        end
    end
end

%% 6. Invert and solve
inverse_Elliptic = inv(Elliptic_spec);
u_flatten        = GF * inverse_Elliptic * f_h;

%% 7. Reshape back
u = real(ipermute(reshape(u_flatten, fliplr(N_vecs)), dim:-1:1));

end
