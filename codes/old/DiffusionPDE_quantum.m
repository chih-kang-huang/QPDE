%% Quantum Generic Diffusion Solver with Detailed Plots
clear; clc; close all;

%% 1. Setup & Parameters
n = 3;              % Qubits per dimension (N = 32)
N = 2^n;
dt = 1e-3;
steps = 100;
dim = 2;            % Dimension of the problem (2D)

% Physical Grid [0, 1)
L = 1;
dx = L / N;
x = (0:N-1) * dx; 
[X, Y] = ndgrid(x, x);

% --- GENERIC SQUARE MATRIX A ---
% A = [3.0, 1.0; 
%      1.0, 2.0];
A=eye(2);

% Check if A is identity matrix
is_identity = all(all(A == eye(size(A))));
if is_identity
    matrix_label = 'I';
else
    matrix_label = 'A';
end

% Source Term (f) and Initial Condition (u)
f = cos(2*pi*X) .* sin(-4*pi*Y);
u_init = cos(2*pi*X) .* sin(8*pi*Y) + 2*sin(6*pi*Y) + 3*sin(10*pi*X).*cos(12*pi*Y).^2;

%% 2. Operator Construction (Pre-computation)
fprintf('Building Quantum Operator for Generic A...\n');

k_indices = [0:N/2-1, -N/2:-1]';
d_eigs = 2i * pi * k_indices / L;

% Create 2D frequency grids
[K_x, K_y] = meshgrid(d_eigs, d_eigs);

% Elliptic operator eigenvalues (not inverted)
Elliptic_diag = A(1,1) * K_x.^2 + A(1,2) * K_x .* K_y + A(2,1) * K_x .* K_y + A(2,2) * K_y.^2;

% Filter (element-wise division in spectral space)
Filter = 1 ./ (1 - dt * Elliptic_diag);

% Quantum Encoding
alpha = max(abs(Filter(:)));     

DiagGate = MakeUnitary(diag(Filter(:) / alpha));

% Circuit: QFT -> Diagonal -> iQFT
FG = GroupFourier(dim, n);               
GF = FG.ctranspose();                  

QC = qclab.QCircuit(dim*n + 1);
QC.push_back(FG);
QC.push_back(qclab.qgates.MatrixGate(0:dim*n, DiagGate, "D"));
QC.push_back(GF);

% Extract Matrix
FullMat = QC.matrix;
Q_Op = FullMat(1:N^dim, 1:N^dim);

%% 3. Time Evolution Loop
u_class = u_init;
u_quant = u_init;

energy_hist_class = zeros(1, steps+1);
energy_hist_quant = zeros(1, steps+1);

calc_energy = @(u_in) 0.5 * real(sum(sum( conj(fft2(u_in)) .* (-Elliptic_diag) .* fft2(u_in) ))) / N^(2*dim);

energy_hist_class(1) = calc_energy(u_class);
energy_hist_quant(1) = calc_energy(u_quant);

fprintf('Running %d steps (N=%d)...\n', steps, N);

for t = 1:steps
    % Classical
    u_h = fftn(u_class);
    f_h = fftn(f);
    u_class = real(ifftn((u_h - dt * f_h) .* Filter));

    % Quantum
    v = u_quant - dt*f;
    u_vec_next = (Q_Op * v(:)) * alpha;
    u_quant = reshape(real(u_vec_next), repmat(N, 1, dim));
    
    energy_hist_class(t+1) = calc_energy(u_class);
    energy_hist_quant(t+1) = calc_energy(u_quant);
end

%% 4. Visualization
diff_norm = norm(u_class(:) - u_quant(:), 'fro');
abs_err_map = abs(u_class - u_quant);
rel_err_map = abs_err_map ./ (abs(u_class) + 1e-10); % Avoid division by zero
energy_err = abs(energy_hist_class - energy_hist_quant);



figure('Position', [50, 50, 1600, 800], 'Name', 'Generic Diffusion Results');

% --- PLOT 1: Energy Evolution ---
subplot(2, 4, 1);
plot(0:steps, energy_hist_class, 'b-', 'LineWidth', 2); hold on;
plot(0:steps, energy_hist_quant, 'r--', 'LineWidth', 2);
title('Energy Evolution');
xlabel('Time Step'); ylabel('Energy');
legend('Classical', 'Quantum'); grid on;

% --- PLOT 2: Absolute Energy Error ---
subplot(2, 4, 2);
semilogy(0:steps, energy_err, 'k-o', 'LineWidth', 1.5, 'MarkerSize', 4);
title('Abs. Energy Error |E_{cl} - E_{qu}|');
xlabel('Time Step'); ylabel('Error'); grid on;

% --- PLOT 3: Classical Solution ---
subplot(2, 4, 3);
imagesc(u_class'); 
axis square; colorbar;
title('Classical Solution');
xlabel('x'); ylabel('y'); set(gca, 'YDir', 'normal');

% --- PLOT 4: Quantum Solution ---
subplot(2, 4, 4);
imagesc(u_quant'); 
axis square; colorbar;
title('Quantum Solution');
xlabel('x'); ylabel('y'); set(gca, 'YDir', 'normal');

% --- PLOT 5: Absolute Error Map ---
subplot(2, 4, 5);
imagesc(abs_err_map'); 
axis square; colorbar;
title(sprintf('Absolute Error\nMax: %.2e', max(abs_err_map(:))));
xlabel('x'); ylabel('y'); set(gca, 'YDir', 'normal');

% --- PLOT 6: Relative Error Map ---
subplot(2, 4, 6);
imagesc(rel_err_map'); 
axis square; colorbar;
title('Relative Error');
xlabel('x'); ylabel('y'); set(gca, 'YDir', 'normal');

% --- PLOT 7: Cross-Section (Middle of Y) ---
subplot(2, 4, 7);
mid_idx = floor(N/2);
plot(x, u_class(:, mid_idx), 'b-', 'LineWidth', 2); hold on;
plot(x, u_quant(:, mid_idx), 'r--', 'LineWidth', 2);
title(sprintf('Cross-section at y=%.2f', x(mid_idx)));
legend('Classical', 'Quantum'); grid on;
xlabel('x'); ylabel('u(x, y_{mid})');

% --- PLOT 8: Error Metrics (Text Box) ---
subplot(2, 4, 8);
axis off;
text(0.1, 0.8, sprintf('Run Metrics (Steps=%d)', steps), 'FontWeight', 'bold', 'FontSize', 12);
text(0.1, 0.6, sprintf('Frobenius Norm: %.3e', diff_norm));
text(0.1, 0.5, sprintf('Max Abs Error: %.3e', max(abs_err_map(:))));
text(0.1, 0.4, sprintf('Mean Abs Error: %.3e', mean(abs_err_map(:))));
text(0.1, 0.3, sprintf('Final Energy (Cl): %.4f', energy_hist_class(end)));
text(0.1, 0.2, sprintf('Final Energy (Qu): %.4f', energy_hist_quant(end)));
text(0.1, 0.1, sprintf('Encoding Alpha: %.4f', alpha));
box on;

% Save main figure as PNG
png_filename_main = sprintf('Heat2D_%s_main.png', matrix_label);
saveas(gcf, png_filename_main);
fprintf('Saved main figure to %s\n', png_filename_main);

%% 5. Additional Visualization Functions (Converted from Python)

% Define domain bounds for visualization
x_lb = 0;
x_rb = L;
y_lb = 0;
y_ub = L;

% --- FIGURE 1: Energy Comparison with Log Scale ---
time_vec = dt * (0:steps);
log_energy_class = energy_hist_class;  % Avoid log(0)
log_energy_quant = energy_hist_quant;

figure('Name', 'Energy Comparison (Log Scale)', 'NumberTitle', 'off');
plot(time_vec, log_energy_class, 'b-', 'LineWidth', 2); hold on;
plot(time_vec, log_energy_quant, 'r--', 'LineWidth', 2);
ylabel('Energy', 'FontSize', 12);
xlabel('Time', 'FontSize', 12);
yscale log
legend('$Energy_{classical}$', '$Energy_{quantum}$', 'Interpreter', 'latex', 'FontSize', 11);
grid on;
set(gca, 'FontSize', 11);

% Save to HDF5 format
h5_filename = sprintf('Heat2D_%s_energy.h5', matrix_label);

% Delete file if it already exists
if isfile(h5_filename)
    delete(h5_filename);
end

% Create file and datasets
h5create(h5_filename, '/time_vector', size(time_vec), 'Datatype', 'double');
h5create(h5_filename, '/log_energy_classical', size(log_energy_class), 'Datatype', 'double');
h5create(h5_filename, '/log_energy_quantum', size(log_energy_quant), 'Datatype', 'double');

% Write data
h5write(h5_filename, '/time_vector', time_vec);
h5write(h5_filename, '/log_energy_classical', log_energy_class);
h5write(h5_filename, '/log_energy_quantum', log_energy_quant);
fprintf('Saved energy data to %s\n', h5_filename);

% Save figure as PNG
png_filename_energy = sprintf('Heat2D_%s_energy.png', matrix_label);
saveas(gcf, png_filename_energy);
fprintf('Saved energy figure to %s\n', png_filename_energy);

% --- FIGURE 2: Solution Comparison and Error Map ---
figure('Name', 'Solution Comparison', 'NumberTitle', 'off');
set(gcf, 'Position', [100, 100, 1400, 400]);

% Subplot 1: Numerical (Classical) Solution
subplot(1, 3, 1);
imagesc([x_lb, x_rb], [y_lb, y_ub], u_class');
set(gca, 'YDir', 'normal');
axis off;
title('u_{numerical}', 'FontSize', 18);
colorbar;

% Subplot 2: Analytical (Quantum) Solution
subplot(1, 3, 2);
imagesc([x_lb, x_rb], [y_lb, y_ub], u_quant');
set(gca, 'YDir', 'normal');
axis off;
title('u_{quantum}', 'FontSize', 18);
colorbar;

% Subplot 3: Absolute Error
subplot(1, 3, 3);
error_field = abs(u_quant - u_class);
imagesc([x_lb, x_rb], [y_lb, y_ub], error_field');
set(gca, 'YDir', 'normal');
title('abs. error', 'FontSize', 18);
colorbar;
axis off;

% Save to HDF5 format
h5_filename_sol = sprintf('Heat2D_%s_num.h5', matrix_label);

% Delete file if it already exists
if isfile(h5_filename_sol)
    delete(h5_filename_sol);
end

% Create file and datasets
h5create(h5_filename_sol, '/u_classical', size(u_class), 'Datatype', 'double');
h5create(h5_filename_sol, '/u_quantum', size(u_quant), 'Datatype', 'double');
h5create(h5_filename_sol, '/error_field', size(error_field), 'Datatype', 'double');
h5create(h5_filename_sol, '/x_coords', size(x), 'Datatype', 'double');
h5create(h5_filename_sol, '/y_coords', size(x), 'Datatype', 'double');
h5create(h5_filename_sol, '/domain_bounds', [1 4], 'Datatype', 'double');

% Write data
h5write(h5_filename_sol, '/u_classical', u_class);
h5write(h5_filename_sol, '/u_quantum', u_quant);
h5write(h5_filename_sol, '/error_field', error_field);
h5write(h5_filename_sol, '/x_coords', x);
h5write(h5_filename_sol, '/y_coords', x);
h5write(h5_filename_sol, '/domain_bounds', [x_lb, x_rb, y_lb, y_ub]);
fprintf('Saved solution data to %s\n', h5_filename_sol);

% Save figure as PNG
png_filename_sol = sprintf('Heat2D_%s_num.png', matrix_label);
saveas(gcf, png_filename_sol);
fprintf('Saved solution figure to %s\n', png_filename_sol);
close;

fprintf('\n========== RESULTS SAVED ==========\n');
fprintf('HDF5 Data Files:\n');
fprintf('  - Heat2D_%s_energy.h5 (Energy evolution data)\n', matrix_label);
fprintf('  - Heat2D_%s_num.h5 (Solution fields and error maps)\n', matrix_label);
fprintf('\nPNG Visualization Files:\n');
fprintf('  - Heat2D_%s_main.png (8-subplot overview)\n', matrix_label);
fprintf('  - Heat2D_%s_energy.png (Energy comparison with log scale)\n', matrix_label);
fprintf('  - Heat2D_%s_num.png (Solution comparison and error map)\n', matrix_label);
fprintf('===================================\n');

diff_norm = norm(u_class - u_quant);
rel_err = diff_norm/norm(u_class); % Avoid division by zero




fprintf('relative error: %.3e\n', rel_err);