%% Quantum Generic Diffusion Solver - Arbitrary Dimension d
clear; clc; close all;

%% 1. Setup & Parameters
n = 3;              % Qubits per dimension (N = 2^n = 8)
N = 2^n;
dt = 1e-3;
steps = 100;
d = 2;              % *** DIMENSION - Change this to desired dimension ***

% Physical Grid [0, 1)
L = 1;
dx = L / N;
x = (0:N-1) * dx;

% --- GENERIC d×d SYMMETRIC POSITIVE DEFINITE MATRIX A ---
% Create a random SPD matrix by A = Q*Lambda*Q' where Q is orthogonal
% and Lambda is diagonal with positive eigenvalues
rng(42);  % Fixed seed for reproducibility
Q_rand = orth(randn(d, d));
lambda = linspace(1, d, d);  % Eigenvalues from 1 to d
A = Q_rand * diag(lambda) * Q_rand';
A = (A + A') / 2;  % Ensure symmetry
A=eye(2)
fprintf('=== Quantum Diffusion Solver (Dimension d=%d) ===\n', d);
fprintf('Matrix A (%dx%d):\n', d, d);
disp(A);

% Create multi-dimensional grid
grid_size = repmat(N, 1, d);
[grids{1:d}] = ndgrid(x);  % Create d grids
X_grids = grids;  % Store for later use

% Source Term and Initial Condition
f = cos(2*pi*X_grids{1});
u_init = cos(2*pi*X_grids{1});
for i = 2:d
    f = f .* sin(2*pi*X_grids{i});
    u_init = u_init + sin((i+1)*pi*X_grids{i});
end

%% 2. Operator Construction (Pre-computation)
fprintf('Building Quantum Operator for Generic %dx%d A...\n', d, d);

% Wave Vectors
k_ind = [0:N/2-1, -N/2:-1]';
k_base = 2i * pi * k_ind;

% Create K grids for all dimensions
[K_grids{1:d}] = ndgrid(k_base);

% Construct Elliptic Operator: sum_{i,j} A(i,j) * K_i * K_j
Elliptic_diag = zeros(grid_size);
for i = 1:d
    for j = 1:d
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
total_qubits = d * n;
FG = GroupFourier(d, n);
GF = FG.ctranspose();

QC = qclab.QCircuit(total_qubits + 1);
QC.push_back(FG);
QC.push_back(qclab.qgates.MatrixGate(0:total_qubits, DiagGate, "D"));
QC.push_back(GF);

% Extract Matrix
FullMat = QC.matrix;
Q_Op = FullMat(1:N^d, 1:N^d);

%% 3. Time Evolution Loop
u_class = u_init;
u_quant = u_init;

energy_hist_class = zeros(1, steps+1);
energy_hist_quant = zeros(1, steps+1);

% Energy calculation function
calc_energy = @(u_in) 0.5 * real(sum(conj(fftn(u_in)) .* (-Elliptic_diag) .* fftn(u_in), 'all')) / N^(2*d);

energy_hist_class(1) = calc_energy(u_class);
energy_hist_quant(1) = calc_energy(u_quant);

fprintf('Running %d steps (N=%d, dimension=%d, total gridpoints=%d)...\n', ...
    steps, N, d, N^d);

for t = 1:steps
    % Classical
    u_h = fftn(u_class);
    f_h = fftn(f);
    u_class = real(ifftn((u_h - dt * f_h) .* Filter));

    % Quantum
    v = u_class - dt*f;
    u_vec_next = (Q_Op * v(:)) * alpha;
    u_quant = reshape(real(u_vec_next), grid_size);
    
    energy_hist_class(t+1) = calc_energy(u_class);
    energy_hist_quant(t+1) = calc_energy(u_quant);
    
    if mod(t, 10) == 0
        fprintf('  Step %d/%d\n', t, steps);
    end
end

%% 4. Compute Error Metrics
diff_norm = norm(u_class(:) - u_quant(:), 'fro');
abs_err_map = abs(u_class - u_quant);
rel_err_map = abs_err_map ./ (abs(u_class) + 1e-10);
energy_err = abs(energy_hist_class - energy_hist_quant);

fprintf('Final Frobenius Error: %.3e\n', diff_norm);
fprintf('Max Absolute Error: %.3e\n', max(abs_err_map(:)));
fprintf('Mean Absolute Error: %.3e\n', mean(abs_err_map(:)));

%% 5. Visualization - Adaptive Based on Dimension

% Define slicing indices
mid_idx = floor(N/2);
edge_idx = 1;

% For dimensions > 3, we'll create slices by fixing all but 2 dimensions
if d <= 3
    create_full_visualization(d, N, n, steps, x, ...
        u_class, u_quant, abs_err_map, ...
        energy_hist_class, energy_hist_quant, energy_err, ...
        diff_norm, alpha, dt, dx, X_grids, K_grids);
else
    create_high_dim_visualization(d, N, n, steps, x, ...
        u_class, u_quant, abs_err_map, ...
        energy_hist_class, energy_hist_quant, energy_err, ...
        diff_norm, alpha, dt, dx, grid_size);
end

fprintf('\n=== Simulation Complete ===\n');
fprintf('Problem Dimension: %d\n', d);
fprintf('Grid Size: %s\n', sprintf(repmat('%d×', 1, d-1) + "%d", grid_size));
fprintf('Total grid points: %d\n', N^d);
fprintf('Total qubits: %d\n', d*n);

%% ========== NESTED FUNCTION: Low-Dimensional Visualization ==========
function create_full_visualization(d, N, n, steps, x, ...
    u_class, u_quant, abs_err_map, ...
    energy_hist_class, energy_hist_quant, energy_err, ...
    diff_norm, alpha, dt, dx, X_grids, K_grids)

mid_idx = floor(N/2);

if d == 1
    % ===== 1D VISUALIZATION =====
    figure('Position', [50, 50, 1400, 700], 'Name', '1D Quantum Diffusion Results');
    
    % Plot 1: Solution
    subplot(2, 3, 1);
    plot(x, u_class, 'b-', 'LineWidth', 2); hold on;
    plot(x, u_quant, 'r--', 'LineWidth', 2);
    title('Solution u(x)');
    xlabel('x'); ylabel('u');
    legend('Classical', 'Quantum'); grid on;
    
    % Plot 2: Absolute Error
    subplot(2, 3, 2);
    semilogy(x, abs_err_map, 'ko-', 'LineWidth', 1.5, 'MarkerSize', 3);
    title('Absolute Error |u_{cl} - u_{qu}|');
    xlabel('x'); ylabel('Error'); grid on;
    
    % Plot 3: Energy Evolution
    subplot(2, 3, 3);
    plot(0:steps, energy_hist_class, 'b-', 'LineWidth', 2); hold on;
    plot(0:steps, energy_hist_quant, 'r--', 'LineWidth', 2);
    title('Energy Evolution');
    xlabel('Time Step'); ylabel('Energy');
    legend('Classical', 'Quantum'); grid on;
    
    % Plot 4: Energy Error
    subplot(2, 3, 4);
    semilogy(0:steps, energy_err, 'k-o', 'LineWidth', 1.5, 'MarkerSize', 4);
    title('Absolute Energy Error');
    xlabel('Time Step'); ylabel('Error'); grid on;
    
    % Plot 5: Relative Error
    subplot(2, 3, 5);
    rel_err = abs_err_map ./ (abs(u_class) + 1e-10);
    semilogy(x, rel_err, 'go-', 'LineWidth', 1.5, 'MarkerSize', 3);
    title('Relative Error');
    xlabel('x'); ylabel('Relative Error'); grid on;
    
    % Plot 6: Metrics
    subplot(2, 3, 6);
    axis off;
    text(0.05, 0.95, '=== Metrics (1D) ===', 'FontWeight', 'bold', 'FontSize', 11, ...
        'VerticalAlignment', 'top');
    text(0.05, 0.85, sprintf('Grid Size: %d', N), 'FontSize', 10, ...
        'VerticalAlignment', 'top');
    text(0.05, 0.77, sprintf('Time Steps: %d', steps), 'FontSize', 10, ...
        'VerticalAlignment', 'top');
    text(0.05, 0.69, sprintf('dt: %.2e', dt), 'FontSize', 10, ...
        'VerticalAlignment', 'top');
    text(0.05, 0.61, sprintf('dx: %.4f', dx), 'FontSize', 10, ...
        'VerticalAlignment', 'top');
    text(0.05, 0.50, '--- Errors ---', 'FontWeight', 'bold', 'FontSize', 10, ...
        'VerticalAlignment', 'top');
    text(0.05, 0.42, sprintf('Frob. Norm: %.3e', diff_norm), 'FontSize', 9, ...
        'VerticalAlignment', 'top');
    text(0.05, 0.35, sprintf('Max Abs Err: %.3e', max(abs_err_map(:))), 'FontSize', 9, ...
        'VerticalAlignment', 'top');
    text(0.05, 0.28, sprintf('Mean Abs Err: %.3e', mean(abs_err_map(:))), 'FontSize', 9, ...
        'VerticalAlignment', 'top');
    text(0.05, 0.20, sprintf('Final E (Cl): %.5f', energy_hist_class(end)), 'FontSize', 9, ...
        'VerticalAlignment', 'top');
    text(0.05, 0.12, sprintf('Final E (Qu): %.5f', energy_hist_quant(end)), 'FontSize', 9, ...
        'VerticalAlignment', 'top');
    text(0.05, 0.04, sprintf('Alpha: %.4f', alpha), 'FontSize', 9, ...
        'VerticalAlignment', 'top');
    box on;
    
elseif d == 2
    % ===== 2D VISUALIZATION =====
    figure('Position', [50, 50, 1600, 800], 'Name', '2D Quantum Diffusion Results');
    
    % Plot 1: Energy Evolution
    subplot(2, 4, 1);
    plot(0:steps, energy_hist_class, 'b-', 'LineWidth', 2); hold on;
    plot(0:steps, energy_hist_quant, 'r--', 'LineWidth', 2);
    title('Energy Evolution');
    xlabel('Time Step'); ylabel('Energy');
    legend('Classical', 'Quantum'); grid on;
    
    % Plot 2: Absolute Energy Error
    subplot(2, 4, 2);
    semilogy(0:steps, energy_err, 'k-o', 'LineWidth', 1.5, 'MarkerSize', 4);
    title('Abs. Energy Error');
    xlabel('Time Step'); ylabel('Error'); grid on;
    
    % Plot 3: Classical Solution
    subplot(2, 4, 3);
    imagesc(x, x, u_class');
    axis square; colorbar;
    title('Classical Solution');
    xlabel('x'); ylabel('y');
    set(gca, 'YDir', 'normal');
    
    % Plot 4: Quantum Solution
    subplot(2, 4, 4);
    imagesc(x, x, u_quant');
    axis square; colorbar;
    title('Quantum Solution');
    xlabel('x'); ylabel('y');
    set(gca, 'YDir', 'normal');
    
    % Plot 5: Absolute Error Map
    subplot(2, 4, 5);
    imagesc(x, x, abs_err_map');
    axis square; colorbar;
    title(sprintf('Absolute Error\nMax: %.2e', max(abs_err_map(:))));
    xlabel('x'); ylabel('y');
    set(gca, 'YDir', 'normal');
    
    % Plot 6: Relative Error Map
    subplot(2, 4, 6);
    rel_err_map = abs_err_map ./ (abs(u_class) + 1e-10);
    imagesc(x, x, rel_err_map');
    axis square; colorbar;
    title('Relative Error');
    xlabel('x'); ylabel('y');
    set(gca, 'YDir', 'normal');
    
    % Plot 7: Cross-Section
    subplot(2, 4, 7);
    plot(x, u_class(:, mid_idx), 'b-', 'LineWidth', 2); hold on;
    plot(x, u_quant(:, mid_idx), 'r--', 'LineWidth', 2);
    title(sprintf('Cross-section at y=%.2f', x(mid_idx)));
    legend('Classical', 'Quantum'); grid on;
    xlabel('x'); ylabel('u');
    
    % Plot 8: Metrics
    subplot(2, 4, 8);
    axis off;
    text(0.05, 0.95, '=== Metrics (2D) ===', 'FontWeight', 'bold', 'FontSize', 11, ...
        'VerticalAlignment', 'top');
    text(0.05, 0.85, sprintf('Grid: %d×%d', N, N), 'FontSize', 10, ...
        'VerticalAlignment', 'top');
    text(0.05, 0.77, sprintf('Steps: %d', steps), 'FontSize', 10, ...
        'VerticalAlignment', 'top');
    text(0.05, 0.69, sprintf('dt: %.2e', dt), 'FontSize', 10, ...
        'VerticalAlignment', 'top');
    text(0.05, 0.50, '--- Errors ---', 'FontWeight', 'bold', 'FontSize', 10, ...
        'VerticalAlignment', 'top');
    text(0.05, 0.42, sprintf('Frob. Norm: %.3e', diff_norm), 'FontSize', 9, ...
        'VerticalAlignment', 'top');
    text(0.05, 0.35, sprintf('Max Abs Err: %.3e', max(abs_err_map(:))), 'FontSize', 9, ...
        'VerticalAlignment', 'top');
    text(0.05, 0.28, sprintf('Mean Abs Err: %.3e', mean(abs_err_map(:))), 'FontSize', 9, ...
        'VerticalAlignment', 'top');
    text(0.05, 0.20, sprintf('Final E (Cl): %.5f', energy_hist_class(end)), 'FontSize', 9, ...
        'VerticalAlignment', 'top');
    text(0.05, 0.12, sprintf('Final E (Qu): %.5f', energy_hist_quant(end)), 'FontSize', 9, ...
        'VerticalAlignment', 'top');
    text(0.05, 0.04, sprintf('Alpha: %.4f', alpha), 'FontSize', 9, ...
        'VerticalAlignment', 'top');
    box on;
    
elseif d == 3
    % ===== 3D VISUALIZATION (Two Figures) =====
    figure('Position', [50, 50, 1600, 900], 'Name', '3D Diffusion - Part 1');
    
    z_slices = [1, floor(N/2), N];
    
    % Figure 1 - Part 1
    subplot(2, 4, 1);
    plot(0:steps, energy_hist_class, 'b-', 'LineWidth', 2); hold on;
    plot(0:steps, energy_hist_quant, 'r--', 'LineWidth', 2);
    title('Energy Evolution');
    xlabel('Time Step'); ylabel('Energy');
    legend('Classical', 'Quantum'); grid on;
    
    subplot(2, 4, 2);
    semilogy(0:steps, energy_err, 'k-o', 'LineWidth', 1.5, 'MarkerSize', 4);
    title('Abs. Energy Error');
    xlabel('Time Step'); ylabel('Error'); grid on;
    
    for slice_idx = 1:2
        z_pos = z_slices(slice_idx);
        
        subplot(2, 4, 2+slice_idx);
        slice_data = u_class(:, :, z_pos);
        imagesc(x, x, slice_data');
        axis square; colorbar;
        title(sprintf('Classical @ z=%.3f', x(z_pos)));
        xlabel('x'); ylabel('y');
        set(gca, 'YDir', 'normal');
        
        subplot(2, 4, 4+slice_idx);
        slice_data = u_quant(:, :, z_pos);
        imagesc(x, x, slice_data');
        axis square; colorbar;
        title(sprintf('Quantum @ z=%.3f', x(z_pos)));
        xlabel('x'); ylabel('y');
        set(gca, 'YDir', 'normal');
        
        subplot(2, 4, 6+slice_idx);
        slice_err = abs_err_map(:, :, z_pos);
        imagesc(x, x, slice_err');
        axis square; colorbar;
        title(sprintf('Error @ z=%.3f\nMax:%.2e', x(z_pos), max(slice_err(:))));
        xlabel('x'); ylabel('y');
        set(gca, 'YDir', 'normal');
    end
    
    % Figure 1 - Part 2
    figure('Position', [50, 1000, 1600, 900], 'Name', '3D Diffusion - Part 2');
    
    subplot(2, 4, 1);
    z_pos = z_slices(3);
    slice_data = u_class(:, :, z_pos);
    imagesc(x, x, slice_data');
    axis square; colorbar;
    title(sprintf('Classical @ z=%.3f (end)', x(z_pos)));
    xlabel('x'); ylabel('y');
    set(gca, 'YDir', 'normal');
    
    subplot(2, 4, 2);
    z_pos = z_slices(3);
    slice_data = u_quant(:, :, z_pos);
    imagesc(x, x, slice_data');
    axis square; colorbar;
    title(sprintf('Quantum @ z=%.3f (end)', x(z_pos)));
    xlabel('x'); ylabel('y');
    set(gca, 'YDir', 'normal');
    
    subplot(2, 4, 3);
    z_pos = z_slices(3);
    slice_err = abs_err_map(:, :, z_pos);
    imagesc(x, x, slice_err');
    axis square; colorbar;
    title(sprintf('Error @ z=%.3f\nMax:%.2e', x(z_pos), max(slice_err(:))));
    xlabel('x'); ylabel('y');
    set(gca, 'YDir', 'normal');
    
    subplot(2, 4, 4);
    axis off;
    text(0.05, 0.95, '=== Metrics (3D) ===', 'FontWeight', 'bold', 'FontSize', 11, ...
        'VerticalAlignment', 'top');
    text(0.05, 0.85, sprintf('Grid: %d×%d×%d', N, N, N), 'FontSize', 10, ...
        'VerticalAlignment', 'top');
    text(0.05, 0.77, sprintf('Steps: %d', steps), 'FontSize', 10, ...
        'VerticalAlignment', 'top');
    text(0.05, 0.50, '--- Errors ---', 'FontWeight', 'bold', 'FontSize', 10, ...
        'VerticalAlignment', 'top');
    text(0.05, 0.42, sprintf('Frob. Norm: %.3e', diff_norm), 'FontSize', 9, ...
        'VerticalAlignment', 'top');
    text(0.05, 0.35, sprintf('Max Abs Err: %.3e', max(abs_err_map(:))), 'FontSize', 9, ...
        'VerticalAlignment', 'top');
    text(0.05, 0.28, sprintf('Mean Abs Err: %.3e', mean(abs_err_map(:))), 'FontSize', 9, ...
        'VerticalAlignment', 'top');
    text(0.05, 0.20, sprintf('Final E (Cl): %.5f', energy_hist_class(end)), 'FontSize', 9, ...
        'VerticalAlignment', 'top');
    text(0.05, 0.12, sprintf('Final E (Qu): %.5f', energy_hist_quant(end)), 'FontSize', 9, ...
        'VerticalAlignment', 'top');
    text(0.05, 0.04, sprintf('Alpha: %.4f', alpha), 'FontSize', 9, ...
        'VerticalAlignment', 'top');
    box on;
    
    % Orthogonal planes
    mid_idx = floor(N/2);
    
    subplot(2, 4, 5);
    slice_yz = squeeze(u_class(mid_idx, :, :));
    imagesc(x, x, slice_yz');
    axis square; colorbar;
    title(sprintf('Classical YZ @ x=%.3f', x(mid_idx)));
    xlabel('y'); ylabel('z');
    set(gca, 'YDir', 'normal');
    
    subplot(2, 4, 6);
    slice_yz = squeeze(u_quant(mid_idx, :, :));
    imagesc(x, x, slice_yz');
    axis square; colorbar;
    title(sprintf('Quantum YZ @ x=%.3f', x(mid_idx)));
    xlabel('y'); ylabel('z');
    set(gca, 'YDir', 'normal');
    
    subplot(2, 4, 7);
    slice_xz = squeeze(u_class(:, mid_idx, :));
    imagesc(x, x, slice_xz');
    axis square; colorbar;
    title(sprintf('Classical XZ @ y=%.3f', x(mid_idx)));
    xlabel('x'); ylabel('z');
    set(gca, 'YDir', 'normal');
    
    subplot(2, 4, 8);
    slice_xz = squeeze(u_quant(:, mid_idx, :));
    imagesc(x, x, slice_xz');
    axis square; colorbar;
    title(sprintf('Quantum XZ @ y=%.3f', x(mid_idx)));
    xlabel('x'); ylabel('z');
    set(gca, 'YDir', 'normal');
end

end

%% ========== NESTED FUNCTION: High-Dimensional Visualization ==========
function create_high_dim_visualization(d, N, n, steps, x, ...
    u_class, u_quant, abs_err_map, ...
    energy_hist_class, energy_hist_quant, energy_err, ...
    diff_norm, alpha, dt, dx, grid_size)

fprintf('\nDimension d=%d is high-dimensional. Creating 2D slice visualizations.\n', d);

figure('Position', [50, 50, 1400, 900], 'Name', sprintf('High-D (%dD) Diffusion Results', d));

% Plot 1: Energy Evolution
subplot(2, 3, 1);
plot(0:steps, energy_hist_class, 'b-', 'LineWidth', 2); hold on;
plot(0:steps, energy_hist_quant, 'r--', 'LineWidth', 2);
title(sprintf('Energy Evolution (d=%d)', d));
xlabel('Time Step'); ylabel('Energy');
legend('Classical', 'Quantum'); grid on;

% Plot 2: Absolute Energy Error
subplot(2, 3, 2);
semilogy(0:steps, energy_err, 'k-o', 'LineWidth', 1.5, 'MarkerSize', 4);
title(sprintf('Abs. Energy Error (d=%d)', d));
xlabel('Time Step'); ylabel('Error'); grid on;

% Create 2D slices by fixing all but first 2 dimensions at middle index
mid_idx = floor(N/2);

% Extract 2D slice by fixing all dimensions >= 3 to middle index
if d == 4
    slice_classical = u_class(:, :, mid_idx, mid_idx);
    slice_quantum = u_quant(:, :, mid_idx, mid_idx);
    slice_error = abs_err_map(:, :, mid_idx, mid_idx);
elseif d == 5
    slice_classical = u_class(:, :, mid_idx, mid_idx, mid_idx);
    slice_quantum = u_quant(:, :, mid_idx, mid_idx, mid_idx);
    slice_error = abs_err_map(:, :, mid_idx, mid_idx, mid_idx);
elseif d == 6
    slice_classical = u_class(:, :, mid_idx, mid_idx, mid_idx, mid_idx);
    slice_quantum = u_quant(:, :, mid_idx, mid_idx, mid_idx, mid_idx);
    slice_error = abs_err_map(:, :, mid_idx, mid_idx, mid_idx, mid_idx);
else
    % For d > 6, just take first 2D slice
    slice_classical = u_class(:, :);
    slice_quantum = u_quant(:, :);
    slice_error = abs_err_map(:, :);
end

% Plot 3: Classical 2D Slice
subplot(2, 3, 3);
imagesc(x, x, slice_classical');
axis square; colorbar;
title(sprintf('Classical (fixed dims 3+ @ center)'));
xlabel('Dim 1'); ylabel('Dim 2');
set(gca, 'YDir', 'normal');

% Plot 4: Quantum 2D Slice
subplot(2, 3, 4);
imagesc(x, x, slice_quantum');
axis square; colorbar;
title(sprintf('Quantum (fixed dims 3+ @ center)'));
xlabel('Dim 1'); ylabel('Dim 2');
set(gca, 'YDir', 'normal');

% Plot 5: Error 2D Slice
subplot(2, 3, 5);
imagesc(x, x, slice_error');
axis square; colorbar;
title(sprintf('Error (fixed dims 3+ @ center)\nMax: %.2e', max(slice_error(:))));
xlabel('Dim 1'); ylabel('Dim 2');
set(gca, 'YDir', 'normal');

% Plot 6: Metrics
subplot(2, 3, 6);
axis off;
grid_str = sprintf(repmat('%d×', 1, d-1) + "%d", grid_size);
text(0.05, 0.95, sprintf('=== Metrics (%dD) ===', d), 'FontWeight', 'bold', 'FontSize', 11, ...
    'VerticalAlignment', 'top');
text(0.05, 0.85, sprintf('Dimension: %d', d), 'FontSize', 10, 'VerticalAlignment', 'top');
text(0.05, 0.77, sprintf('Grid: %s', grid_str), 'FontSize', 9, 'VerticalAlignment', 'top');
text(0.05, 0.69, sprintf('Points: %d', N^d), 'FontSize', 9, 'VerticalAlignment', 'top');
text(0.05, 0.61, sprintf('Qubits: %d', d*n), 'FontSize', 9, 'VerticalAlignment', 'top');
text(0.05, 0.53, sprintf('Steps: %d', steps), 'FontSize', 9, 'VerticalAlignment', 'top');
text(0.05, 0.45, sprintf('dt: %.2e', dt), 'FontSize', 9, 'VerticalAlignment', 'top');
text(0.05, 0.37, sprintf('dx: %.4f', dx), 'FontSize', 9, 'VerticalAlignment', 'top');
text(0.05, 0.27, '--- Errors ---', 'FontWeight', 'bold', 'FontSize', 10, ...
    'VerticalAlignment', 'top');
text(0.05, 0.19, sprintf('Frob. Norm: %.3e', diff_norm), 'FontSize', 9, ...
    'VerticalAlignment', 'top');
text(0.05, 0.12, sprintf('Max Err: %.3e', max(abs_err_map(:))), 'FontSize', 9, ...
    'VerticalAlignment', 'top');
text(0.05, 0.05, sprintf('Mean Err: %.3e', mean(abs_err_map(:))), 'FontSize', 9, ...
    'VerticalAlignment', 'top');
box on;

end