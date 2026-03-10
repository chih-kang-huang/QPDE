function E = energy(u, A, N, dx, d)
% Calculates Energy using spectral gradients.

    % 1. Forward FFT
    u_h = fftn(u);

    % 2. Generate Wave Numbers
    k_vecs = cell(1, d);
    for dim_i = 1:d
        freqs = [0:N/2-1, -N/2:-1]; 
        % Scale by 2*pi*i / (N*dx)
        k_vals = (2i * pi * freqs) / (N * dx);
        k_vecs{dim_i} = k_vals;
    end

    [K_grids{1:d}] = ndgrid(k_vecs{:});

    % 3. Compute Gradients in Fourier Space
    u_vec_h = cell(1, d);
    for i = 1:d
        u_vec_h{i} = K_grids{i} .* u_h;
    end

    % 4. Compute A * u_vec_h
    Au_vec_h = cell(1, d);
    for i = 1:d
        Au_vec_h{i} = zeros(size(u_h)); 
        for j = 1:d
            if A(i,j) ~= 0
                Au_vec_h{i} = Au_vec_h{i} + A(i,j) * u_vec_h{j};
            end
        end
    end

    % 5. Dot Product (Sum over dimensions)
    dot_product_map = zeros(size(u_h));
    for i = 1:d
        dot_product_map = dot_product_map + conj(u_vec_h{i}) .* Au_vec_h{i};
    end

    % 6. Return Mean Real Part / 2
    E = real(mean(dot_product_map(:))) / 2;
end