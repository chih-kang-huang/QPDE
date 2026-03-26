function [u_class, u_quant, E_class, E_quant] = runTimeEvolution(u_init, f_vals, grids, A, N_vecs, dx, dt, steps, Q_Op, d, N)
    u_class = u_init;
    u_quant = u_init;
    
    E_class = zeros(1, steps);
    E_quant = zeros(1, steps);
    %fprintf('Running %d steps...\n', steps);
    
    for t = 1:steps
        % --- Classical Step ---
        rhs = u_class - dt * f_vals;
        u_class = solver_Diffusion_generic(rhs, grids, A, N_vecs, dx, dt);
        
        % --- Quantum Step ---
        v = u_quant - dt * f_vals; 
        u_vec_next = (Q_Op * v(:));
        u_quant = reshape(real(u_vec_next), N_vecs);
        
        % --- Compute Energy ---
        E_class(t) = energy(u_class, A, N, dx, d);
        E_quant(t) = energy(u_quant, A, N, dx, d);
    end
end