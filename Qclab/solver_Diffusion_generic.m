function u = solver_Diffusion_generic(f, N_vecs, dim, Nx, denom_class)
    %   Spectral solver for (I - dt * div(A grad)) u = f
    %   denom_class: pre-computed 1 ./ (1 - dt * L_h), permuted and flattened
    dfmtx = fft(eye(Nx));
    GF_1d = dfmtx' / Nx;
    FG = dfmtx;
    GF = GF_1d;
    for k = 2:dim
        FG = kron(FG, dfmtx);
        GF = kron(GF, GF_1d);
    end

    f_values  = reshape(f, N_vecs);
    f_flatten = reshape(permute(f_values, dim:-1:1), [], 1);
    f_h       = FG * f_flatten;
    u_h       = f_h .* denom_class;   
    u_flatten = GF * u_h;
    u = real(ipermute(reshape(u_flatten, fliplr(N_vecs)), dim:-1:1));
end