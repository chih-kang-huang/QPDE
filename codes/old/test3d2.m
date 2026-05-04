%% Quantum Generic Diffusion Solver - 3D Version with Detailed Plots
clear; clc; close all;

%% 1. Setup & Parameters
n = 3;              % Qubits per dimension (N = 8)
N = 2^n;
dt = 1e-3;
steps = 100;
dim = 3;            % Dimension of the problem (3D)

% Physical Grid [0, 1)
L = 1;
dx = L / N;
x = (0:N-1) * dx; 
[X, Y, Z] = ndgrid(x, x, x);

% --- GENERIC 3x3 MATRIX A (symmetric positive definite) ---
A = [4.0, 0.5, 0.3; 
     0.5, 3.0, 0.2;
     0.3, 0.2, 2.5];

% Source Term (f) and Initial Condition (u)
f = cos(2*pi*X) .* sin(2*pi*Y) .* cos(2*pi*Z);
u_init = cos(2*pi*X) .* sin(4*pi*Y) .* cos(3*pi*Z) + ...
         2*sin(3*pi*X) .* cos(2*pi*Y) + ...
         sin(2*pi*Z) .* cos(2*pi*X);

%% 2. Operator Construction (Pre-computation)
fprintf('Building Quantum Operator for Generic 3x3 A...\n');

% Wave Vectors
k_ind = [0:N/2-1, -N/2:-1]';
k_base = 2i * pi * k_ind;       
[K_grids{1:dim}] = ndgrid(k_base);

% Construct Elliptic Operator
Elliptic_diag = zeros(size(K_grids{1}));
for i = 1:dim
    for j = 1:dim
        Elliptic_diag = Elliptic_diag + A(i,j) * K_grids{i} .* K_grids{j};
    end
end

% Create Filter
Denominator = 1 - dt * Elliptic_diag;
Filter = 1 ./ Denominator;

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

calc_energy = @(u_in) 0.5 * real(sum(sum(sum( conj(fftn(u_in)) .* (-Elliptic_diag) .* fftn(u_in) )))) / N^(2*dim);

energy_hist_class(1) = calc_energy(u_class);
energy_hist_quant(1) = calc_energy(u_quant);

fprintf('Running %d steps (N=%d, total gridpoints=%d)...\n', steps, N, N^dim);

for t = 1:steps
    % Classical
    u_h = fftn(u_class);
    f_h = fftn(f);
    u_class = real(ifftn((u_h - dt * f_h) .* Filter));

    % Quantum
    v = u_class - dt*f;
    u_vec_next = (Q_Op * v(:)) * alpha;
    u_quant = reshape(real(u_vec_next), repmat(N, 1, dim));
    
    energy_hist_class(t+1) = calc_energy(u_class);
    energy_hist_quant(t+1) = calc_energy(u_quant);
end

%% 4. Compute Error Metrics
diff_norm = norm(u_class(:) - u_quant(:), 'fro');
abs_err_map = abs(u_class - u_quant);
rel_err_map = abs_err_map ./ (abs(u_class) + 1e-10);
energy_err = abs(energy_hist_class - energy_hist_quant);

fprintf('Final Frobenius Error: %.3e\n', diff_norm);
fprintf('Max Absolute Error: %.3e\n', max(abs_err_map(:)));
fprintf('Mean Absolute Error: %.3e\n', mean(abs_err_map(:)));

%% 5. Visualization - Multiple Slices (Two Figures)
figure('Position', [50, 50, 1600, 900], 'Name', '3D Quantum Diffusion Results - Part 1');

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

% --- PLOTS 3-4: Slices at Different Z Levels (Classical) ---
mid_z = floor(N/2);
z_slices = [1, mid_z, N];

subplot(2, 4, 3);
z_pos = z_slices(1);
slice_data = u_class(:, :, z_pos);
imagesc(x, x, slice_data');
axis square; colorbar;
title(sprintf('Classical @ z=%.3f', x(z_pos)));
xlabel('x'); ylabel('y'); 
set(gca, 'YDir', 'normal');

subplot(2, 4, 4);
z_pos = z_slices(2);
slice_data = u_class(:, :, z_pos);
imagesc(x, x, slice_data');
axis square; colorbar;
title(sprintf('Classical @ z=%.3f (mid)', x(z_pos)));
xlabel('x'); ylabel('y'); 
set(gca, 'YDir', 'normal');

% --- PLOTS 5-6: Slices at Different Z Levels (Quantum) ---
subplot(2, 4, 5);
z_pos = z_slices(1);
slice_data = u_quant(:, :, z_pos);
imagesc(x, x, slice_data');
axis square; colorbar;
title(sprintf('Quantum @ z=%.3f', x(z_pos)));
xlabel('x'); ylabel('y'); 
set(gca, 'YDir', 'normal');

subplot(2, 4, 6);
z_pos = z_slices(2);
slice_data = u_quant(:, :, z_pos);
imagesc(x, x, slice_data');
axis square; colorbar;
title(sprintf('Quantum @ z=%.3f (mid)', x(z_pos)));
xlabel('x'); ylabel('y'); 
set(gca, 'YDir', 'normal');

% --- PLOTS 7-8: Error Maps ---
subplot(2, 4, 7);
z_pos = z_slices(1);
slice_err = abs_err_map(:, :, z_pos);
imagesc(x, x, slice_err');
axis square; colorbar;
title(sprintf('Error @ z=%.3f\nMax:%.2e', x(z_pos), max(slice_err(:))));
xlabel('x'); ylabel('y'); 
set(gca, 'YDir', 'normal');

subplot(2, 4, 8);
z_pos = z_slices(2);
slice_err = abs_err_map(:, :, z_pos);
imagesc(x, x, slice_err');
axis square; colorbar;
title(sprintf('Error @ z=%.3f\nMax:%.2e', x(z_pos), max(slice_err(:))));
xlabel('x'); ylabel('y'); 
set(gca, 'YDir', 'normal');

% Second figure for additional slices and metrics
figure('Position', [50, 1000, 1600, 900], 'Name', '3D Quantum Diffusion Results - Part 2');

% --- PLOT 1: Classical at top slice ---
subplot(2, 4, 1);
z_pos = z_slices(3);
slice_data = u_class(:, :, z_pos);
imagesc(x, x, slice_data');
axis square; colorbar;
title(sprintf('Classical @ z=%.3f (end)', x(z_pos)));
xlabel('x'); ylabel('y'); 
set(gca, 'YDir', 'normal');

% --- PLOT 2: Quantum at top slice ---
subplot(2, 4, 2);
z_pos = z_slices(3);
slice_data = u_quant(:, :, z_pos);
imagesc(x, x, slice_data');
axis square; colorbar;
title(sprintf('Quantum @ z=%.3f (end)', x(z_pos)));
xlabel('x'); ylabel('y'); 
set(gca, 'YDir', 'normal');

% --- PLOT 3: Error at top slice ---
subplot(2, 4, 3);
z_pos = z_slices(3);
slice_err = abs_err_map(:, :, z_pos);
imagesc(x, x, slice_err');
axis square; colorbar;
title(sprintf('Error @ z=%.3f\nMax:%.2e', x(z_pos), max(slice_err(:))));
xlabel('x'); ylabel('y'); 
set(gca, 'YDir', 'normal');

% --- PLOT 4: Error Metrics (Text Box) ---
subplot(2, 4, 4);
axis off;
text(0.05, 0.95, '=== Run Metrics ===', 'FontWeight', 'bold', 'FontSize', 11, 'VerticalAlignment', 'top');
text(0.05, 0.85, sprintf('Problem Dim: 3D'), 'FontSize', 10, 'VerticalAlignment', 'top');
text(0.05, 0.77, sprintf('Grid Size: %d×%d×%d', N, N, N), 'FontSize', 10, 'VerticalAlignment', 'top');
text(0.05, 0.69, sprintf('Time Steps: %d', steps), 'FontSize', 10, 'VerticalAlignment', 'top');
text(0.05, 0.61, sprintf('dt: %.2e', dt), 'FontSize', 10, 'VerticalAlignment', 'top');
text(0.05, 0.53, sprintf('dx: %.4f', dx), 'FontSize', 10, 'VerticalAlignment', 'top');
text(0.05, 0.45, '--- Errors ---', 'FontWeight', 'bold', 'FontSize', 10, 'VerticalAlignment', 'top');
text(0.05, 0.37, sprintf('Frob. Norm: %.3e', diff_norm), 'FontSize', 9, 'VerticalAlignment', 'top');
text(0.05, 0.31, sprintf('Max Abs Err: %.3e', max(abs_err_map(:))), 'FontSize', 9, 'VerticalAlignment', 'top');
text(0.05, 0.25, sprintf('Mean Abs Err: %.3e', mean(abs_err_map(:))), 'FontSize', 9, 'VerticalAlignment', 'top');
text(0.05, 0.17, sprintf('Final E (Cl): %.5f', energy_hist_class(end)), 'FontSize', 9, 'VerticalAlignment', 'top');
text(0.05, 0.11, sprintf('Final E (Qu): %.5f', energy_hist_quant(end)), 'FontSize', 9, 'VerticalAlignment', 'top');
text(0.05, 0.05, sprintf('Alpha: %.4f', alpha), 'FontSize', 9, 'VerticalAlignment', 'top');
box on;

% --- PLOTS 5-8: Slices through different axes at center ---
mid_idx = floor(N/2);

% YZ plane (x = mid)
subplot(2, 4, 5);
slice_yz_cl = squeeze(u_class(mid_idx, :, :));
imagesc(x, x, slice_yz_cl');
axis square; colorbar;
title(sprintf('Classical YZ @ x=%.3f', x(mid_idx)));
xlabel('y'); ylabel('z');
set(gca, 'YDir', 'normal');

subplot(2, 4, 6);
slice_yz_qu = squeeze(u_quant(mid_idx, :, :));
imagesc(x, x, slice_yz_qu');
axis square; colorbar;
title(sprintf('Quantum YZ @ x=%.3f', x(mid_idx)));
xlabel('y'); ylabel('z');
set(gca, 'YDir', 'normal');

% XZ plane (y = mid)
subplot(2, 4, 7);
slice_xz_cl = squeeze(u_class(:, mid_idx, :));
imagesc(x, x, slice_xz_cl');
axis square; colorbar;
title(sprintf('Classical XZ @ y=%.3f', x(mid_idx)));
xlabel('x'); ylabel('z');
set(gca, 'YDir', 'normal');

subplot(2, 4, 8);
slice_xz_qu = squeeze(u_quant(:, mid_idx, :));
imagesc(x, x, slice_xz_qu');
axis square; colorbar;
title(sprintf('Quantum XZ @ y=%.3f', x(mid_idx)));
xlabel('x'); ylabel('z');
set(gca, 'YDir', 'normal');

fprintf('\n=== Simulation Complete ===\n');
fprintf('Matrix A (3x3):\n');
disp(A);
fprintf('Total grid points: %d\n', N^dim);
fprintf('Total qubits: %d\n', dim*n);