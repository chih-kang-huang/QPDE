function [u_class, u_quant, u_gt, E_class, E_quant, E_gt] = ...
        runTimeEvolution(u_init, f_vals, A, N_vecs, dx, dt, steps, Q_Op, d,flagUtrue)

    u_class  = u_init;
    u_quant  = u_init;
    N        = N_vecs(1);
    flagGT   = (nargin >= 10);

    E_class  = zeros(1, steps);
    E_quant  = zeros(1, steps);
    E_gt     = zeros(1, steps);

    if flagGT
        u_gt = u_init;
    else
        u_gt = [];
    end

    for t = 1:steps
        % --- Classical Step ---
        rhs     = u_class - dt * f_vals;
        u_class = solver_Diffusion_generic(rhs, A, N_vecs, dx, dt, d,N);

        % --- Quantum Step ---
        v          = u_quant - dt * f_vals;
        u_vec_next = Q_Op * v(:);
        u_quant    = reshape(real(u_vec_next), N_vecs);

        % --- Ground Truth Step ---
        if flagGT
            rhs_gt = u_gt - dt * f_vals;
            u_gt   = computeDiffusionGroundTruth(rhs_gt, A, N_vecs, dx,dt, d);
            E_gt(t) = energy(u_gt, A, N, dx, d);
        end

        % --- Energy ---
        E_class(t) = energy(u_class, A, N, dx, d);
        E_quant(t) = energy(u_quant, A, N, dx, d);
    end
end