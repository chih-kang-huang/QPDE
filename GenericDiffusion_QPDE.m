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

%% --- Operator Construction ---
% fprintf('Building Quantum Operator for d=%d, N=%d...\n', d, N);
Q_Op  = QPDE_Generator_Diffusion(A, params.n, dt);

%% --- Time Evolution Loop ---
[u_class, u_quant, E_class_history, E_quant_history] = runTimeEvolution(...
    u_init, f_vals, grids, A, N_vecs, dx, dt, steps, Q_Op, d, N);

%% --- Ground Truth (spectral, optional) ---
if flagUtrue
    f_an_vals    = f_analytical(grids{:});
    ground_truth = computeSpectralGroundTruth(f_an_vals, A, N, d, dx);
end

%% --- Visualize ---
if flagUtrue
    visualize_simulation_results(u_class, u_quant, d, N, ground_truth);
else
    visualize_simulation_results(u_class, u_quant, d, N);
end
visualize_energy(E_class_history, E_quant_history, dt);

%% --- Save Results ---
if flagUtrue
    saveResults_Diffusion(u_class, u_quant, E_class_history, E_quant_history, d, true, ground_truth);
else
    saveResults_Diffusion(u_class, u_quant, E_class_history, E_quant_history, d, false, []);
end
%% --- Report Errors & Energy ---
if flagUtrue
    reportMetrics(u_class, u_quant, E_class_history, E_quant_history, true, ground_truth);
else
    reportMetrics(u_class, u_quant, E_class_history, E_quant_history, false, []);
end

end