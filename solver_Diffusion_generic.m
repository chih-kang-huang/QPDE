function u = solver_Diffusion_generic(f, A, N_vecs, dx, dt, dim, Nx)
    %   Spectral solver for (I - dt * div(A grad)) u = f
 
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
    

    f_h = FG * f_flatten;

    denom_grid = buildDiffusionDenom(A, N_vecs, dx, dim);

    denom_flatten = reshape(permute(denom_grid, dim:-1:1), [], 1);
    
    Diffusion_spec_diag = 1 - dt * denom_flatten;
    
    u_h = f_h ./ Diffusion_spec_diag;

    u_flatten = GF * u_h;

    u = real(ipermute(reshape(u_flatten, fliplr(N_vecs)), dim:-1:1));
end