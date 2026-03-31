function [grids, N_dir, dx] = GenericHelmoltz_QPDE(f, k, N, d, u_true)

%% --- Setup ---
flagUtrue = (nargin >= 5);
params    = getParams(N, d);
dx        = params.dx;
N_dir     = params.N_dir;
grids     = params.grids;

%% --- Evaluate RHS and Solve ---
f_vals  = f(grids{:});
f_flat  = f_vals(:);

u_generic = solver_Helmoltz_generic(f_flat, grids, k, N_dir, dx);
op = QPDE_Generator_helmoltz(k,d, params.n);
u_quantum = op * f_flat;

%% --- Ground Truth (spectral, optional) ---
if flagUtrue
    ground_truth = computeSpectralGroundTruthHelmoltz(f_vals, k, N, d, dx);
end

%% --- Visualize ---
visualize_simulation_results(u_generic, real(u_quantum), d, N,ground_truth)

%% --- Save Results ---
saveResults(u_generic, u_quantum, d, flagUtrue, ground_truth);

%% --- Report Errors ---
u_q_real = real(u_quantum);
if flagUtrue
    rel_err_classical = norm(u_generic(:)  - ground_truth(:)) / norm(ground_truth(:));
    rel_err_quantum   = norm(u_q_real(:)   - ground_truth(:)) / norm(ground_truth(:));
    fprintf('\n--- Relative Errors (vs Ground Truth) ---\n');
    fprintf('Classical Relative Error: %.4e\n', rel_err_classical);
    fprintf('Quantum   Relative Error: %.4e\n', rel_err_quantum);
else
    rel_err_quantum = norm(u_q_real(:) - u_generic(:)) / norm(u_generic(:));
    fprintf('\n--- Relative Error (vs Classical Solution) ---\n');
    fprintf('Quantum Relative Error: %.4e\n', rel_err_quantum);
end

end