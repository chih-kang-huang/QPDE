function [grids, N_dir, dx] = GenericElliptic_QPDE(f, A, N, d, u_true)

%% --- Setup ---
flagUtrue = (nargin >= 5);
params    = getParams(N, d);
dx        = params.dx;
N_dir     = params.N_dir;
grids     = params.grids;

%% --- Evaluate RHS and Solve ---
f_vals  = f(grids{:});
f_flat  = f_vals(:);

u_generic = solver_Elliptic_generic(f_flat, grids, A, N_dir, dx);

op = QPDE_Generator(A, params.n);
u_quantum = op * f_flat;

%% --- Ground Truth (spectral, optional) ---
if flagUtrue
    % ground_truth = computeSpectralGroundTruth(f_vals, A, N, d, dx);
    ground_truth=u_true(grids{:});
end

%% --- Visualize ---
visualize_simulation_results(u_generic, real(u_quantum), d, N,ground_truth)

%% --- Save Results ---
saveResults(u_generic, u_quantum, d, flagUtrue, ground_truth);

%% --- Report Errors ---
u_q_real = real(u_quantum);
if flagUtrue
% Absolute Errors
    abs_err_classical = norm(u_generic(:)  - ground_truth(:));
    abs_err_quantum   = norm(u_q_real(:)   - ground_truth(:));

    % Relative Errors
    rel_err_classical = abs_err_classical / norm(ground_truth(:));
    rel_err_quantum   = abs_err_quantum   / norm(ground_truth(:));

    fprintf('\n--- Absolute Errors (vs Ground Truth) ---\n');
    fprintf('Classical Absolute Error: %.4e\n', abs_err_classical);
    fprintf('Quantum   Absolute Error: %.4e\n', abs_err_quantum);

    % fprintf('\n--- Relative Errors (vs Ground Truth) ---\n');
    fprintf('Classical Relative Error: %.4e\n', rel_err_classical);
    fprintf('Quantum   Relative Error: %.4e\n', rel_err_quantum);
else
    rel_err_quantum = norm(u_q_real(:) - u_generic(:)) / norm(u_generic(:));
    fprintf('\n--- Relative Error (vs Classical Solution) ---\n');
    fprintf('Quantum Relative Error: %.4e\n', rel_err_quantum);
end

end