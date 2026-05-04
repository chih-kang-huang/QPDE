function [u_class, u_quant, u_gt, E_class, E_quant, E_gt] = ...
        runTimeEvolution(u_init, f_vals, A, N_vecs, dx, dt, steps, Q_Op, d, flagUtrue)
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

    ground_truth = computeSpectralGroundTruth(f_vals, A, N, d, dx);
    E_ref = new_energy(ground_truth, f_vals, A, N, dx, d);

    % --- Pre-compute spectral denominator for ground truth solver ---
    L_h=buildDiffusionDenom(A, N_vecs, dx, dim);

    
    denom_gt = 1 ./ (1 - dt * L_h);
    fprintf("cond of denom (4.8): %f \n",cond(diag(denom_gt(:)),'inf'))

    denom_class = 1 ./ (1 - dt * reshape(permute(L_h, d:-1:1), [], 1));

    for t = 1:steps
        % --- Classical Step ---
        rhs     = u_class - dt * f_vals;
        u_class = solver_Diffusion_generic(rhs, N_vecs, d, N, denom_class);

        % --- Quantum Step ---
        v          = u_quant - dt * f_vals;
        u_vec_next = Q_Op * v(:);
        u_quant    = reshape(real(u_vec_next), N_vecs);

        % --- Ground Truth Step ---
        if flagGT
            rhs_gt = u_gt - dt * f_vals;
             u_gt   = computeDiffusionGroundTruth(rhs_gt, denom_gt);
            E_gt(t) = new_energy(u_gt, f_vals, A, N, dx, d) - E_ref + 1e-6;
        end

        % --- Energy ---
        E_class(t) = new_energy(u_class, f_vals, A, N, dx, d) - E_ref + 1e-6;
        E_quant(t) = new_energy(u_quant, f_vals, A, N, dx, d) - E_ref + 1e-6;
    end
end

