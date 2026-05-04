function denom = buildEllipticDenom(A, N_vecs, dx, dim)

    % Get 1D eigenvalue vectors for each dimension
    k_vecs = cell(1, dim);
    for d = 1:dim
        L          = N_vecs(d) * dx;
        k_vecs{d}  = spectral_eigenvalues(N_vecs(d), false,L);
    end

    % Expand to N-D grids
    K_grids = cell(1, dim);
    [K_grids{:}] = ndgrid(k_vecs{:});

    % Accumulate A_ij * k_i * k_j
    denom = zeros(N_vecs);
    for i = 1:dim
        for j = 1:dim
            if A(i,j) ~= 0
                denom = denom + A(i,j) * K_grids{i} .* K_grids{j};
            end
        end
    end
denom(1,1)=1;

end