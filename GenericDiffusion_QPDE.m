function GenericDiffusion_QPDE(f_handle, u_init_handle, A, N, d, dt, steps, f_analytical)
% GENERICDIFFUSION_QPDE Generic Quantum Diffusion Solver (2D/3D).
% Uses implicit time-stepping for classical verification.

%% --- Setup ---
flagUtrue = (nargin >= 8);
params    = getParams(N, d);
dx        = params.dx;
N_vecs    = params.N_dir;
grids     = params.grids;

%% --- Evaluate Source and Initial Condition ---
f_vals = f_handle(grids{:});
u_init = u_init_handle(grids{:});

% %% --- Ground Truth (spectral steady-state, optional) ---
% if flagUtrue
%     ground_truth = computeDiffusionGroundTruth(f_vals, A, N_vecs, dx, d);
% else
%     ground_truth = [];
% end

%% --- Operator Construction ---
Q_Op = QPDE_Generator_Diffusion(A, params.n, dt);

%% --- Time Evolution Loop ---
[u_class, u_quant, u_gt, E_class_history, E_quant_history, E_gt_history] = runTimeEvolution(...
    u_init, f_vals, A, N_vecs, dx, dt, steps, Q_Op, d,flagUtrue);%ground_truth

%% --- Visualize ---
if flagUtrue
    visualize_simulation_results(u_class, u_quant, d, N, u_gt);
else
    visualize_simulation_results(u_class, u_quant, d, N);
end

if flagUtrue
    visualize_energy(E_class_history, E_quant_history, E_gt_history, dt);
else
    visualize_energy(E_class_history, E_quant_history, dt);
end


%% --- Save Results ---
if flagUtrue
    saveResults_Diffusion(u_class, u_quant, E_class_history, E_quant_history, d, true, u_gt, E_gt_history);
else
    saveResults_Diffusion(u_class, u_quant, E_class_history, E_quant_history, d, false, [], []);
end

% --- Report Errors & Energy ---
if flagUtrue
    reportMetrics(u_class, u_quant, E_class_history, E_quant_history, true, u_gt, E_gt_history);
else
    reportMetrics(u_class, u_quant, E_class_history, E_quant_history, false, [], []);
end

end