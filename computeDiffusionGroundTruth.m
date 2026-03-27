function ground_truth = computeDiffusionGroundTruth(rhs, A, N_vecs, dx, dt, dim)
    % 1. Transform the pre-calculated RHS to spectral domain
    % Based on your loop: rhs = u_gt - dt * f_vals
    rhs_h = fftn(rhs);
    
    % 2. Generate spectral frequencies for each dimension (i * xi)
    % Calling spectral_eigenvalues with setone = false to keep k(1) = 0
    k_vecs = cell(1, dim);
    for i = 1:dim
        L = N_vecs(i) * dx;
        % Signature: spectral_eigenvalues(N, setone, L)
        k_vecs{i} = spectral_eigenvalues(N_vecs(i), false, L); 
    end
    
    % 3. Create frequency grids (equivalent to np.meshgrid indexing='ij')
    K_grids = cell(1, dim);
    [K_grids{1:dim}] = ndgrid(k_vecs{:});
    
    % 4. Build the Diffusion Symbol (L_h)
    % This represents the spectral multiplier for: div(A grad)
    % L_h = sum_{i,j} A_ij * k_i * k_j
    L_h = zeros(size(rhs_h));
    for i = 1:dim
        for j = 1:dim
            if A(i,j) ~= 0
                % k is imaginary, so k_i * k_j is real/negative on diagonal
                L_h = L_h + A(i,j) .* K_grids{i} .* K_grids{j};
            end
        end
    end
    
    % 5. Build the Implicit Operator Denominator: (I - dt * L_h)
    % This corresponds to the (ones - dt * diffuse_u_h) in your Python code
    denom = 1 - dt * L_h;
    
    % 6. Apply inverse operator and transform back to spatial domain
    % ground_truth = (I - dt*L)^-1 * rhs
    ground_truth = real(ifftn(rhs_h ./ denom));
end