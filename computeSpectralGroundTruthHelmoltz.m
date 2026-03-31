function ground_truth = computeSpectralGroundTruthHelmoltz(f_vals, k, N, d, dx)
    f_h    = fftn(f_vals);
    L      = N * dx;
    k_vecs = cell(1, d);
    for i = 1:d
        k_vecs{i} = spectral_eigenvalues(N, true, L);  % k(1)=1 by default
    end
    K_grids = cell(1, d);
    [K_grids{1:d}] = ndgrid(k_vecs{:});
    lambda = zeros(size(f_h));
    A=eye(d);
    for i = 1:d
        for j = 1:d
            lambda = lambda + A(i,j) .* K_grids{i} .* K_grids{j};
        end
    end
    lambda=lambda+(eye(size(lambda))*k);
    ground_truth = real(ifftn(f_h ./ lambda));
end