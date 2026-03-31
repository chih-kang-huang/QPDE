function E = new_energy(u, f, A, N, dx, d)
% Calculates Energy using spectral gradients in physical space.

    u_h = fftn(u);
    
    k_vecs = cell(1, d);
    for dim_i = 1:d
        freqs = [0:N/2-1, -N/2:-1]; 
        k_vecs{dim_i} = (2i * pi * freqs) / (N * dx);
    end
    [K_grids{1:d}] = ndgrid(k_vecs{:});
    
   
    u_grad = zeros([size(u), d]); 
    idx = repmat({':'}, 1, d); 
    for i = 1:d
       
        grad_i = real(ifftn(K_grids{i} .* u_h));
        u_grad(idx{:}, i) = grad_i; 
    end
    
  
    num_pixels = numel(u);
    
  
    u_grad_flat = reshape(u_grad, num_pixels, d); 
    
  
    Au_vec_flat = (A * u_grad_flat.').'; 
    
   
    dot_product_flat = dot(Au_vec_flat, u_grad_flat, 2);
    
   
    E = mean(dot_product_flat / 2 + f(:) .* u(:));
end