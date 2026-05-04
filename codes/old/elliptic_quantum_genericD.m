%% Generic d-Dimensional Spectral Elliptic Solver
% Solves: L[u] = f where L is an elliptic operator with generic d×d matrix A
% Uses spectral methods (FFT-based) with quantum circuit encoding
% Works for arbitrary dimensions d

clear; clc; close all;

%% 1. PARAMETERS
n = 3;              % Qubits per dimension (N = 2^n)
N = 2^n;
d = 2;              % *** DIMENSION - Change this ***

% Domain
x_lb = 0;
x_rb = 1;
L = x_rb - x_lb;
dx = L / N;
x = x_lb + (0:N-1) * dx;

fprintf('=== Generic %dD Spectral Elliptic Solver ===\n', d);
fprintf('Grid: N=%d per dimension (2^%d)\n', N, n);
fprintf('Total grid points: %d\n', N^d);

%% 2. CREATE d×d COEFFICIENT MATRIX A (Symmetric Positive Definite)
% Generate random SPD matrix
rng(42);
Q_rand = orth(randn(d, d));
lambda = linspace(1, d, d);
A = Q_rand * diag(lambda) * Q_rand';
A = (A + A') / 2;  % Ensure symmetry
A=eye(d);
fprintf('Matrix A (%dx%d):\n', d, d);
disp(A);

%% 3. CREATE d-DIMENSIONAL GRID
grid_size = repmat(N, 1, d);
[grids{1:d}] = ndgrid(x);

fprintf('Grid size: %s\n', sprintf(repmat('%d×', 1, d-1) + "%d", grid_size));

%% 4. SPECTRAL DERIVATIVE OPERATOR
% Wave vector: k_n = 2π n / L, n = 0, 1, ..., N/2-1, -N/2, ..., -1
k = 2*pi/L * [0:N/2-1, -N/2:-1]';
D = diag(1i * k);  % First derivative operator in Fourier space

fprintf('Building elliptic operator...\n');

%% 5. BUILD ELLIPTIC OPERATOR USING KRONECKER PRODUCTS
% For d dimensions: L = Σᵢⱼ A(i,j) × ∂²/∂x_i∂x_j
% In Fourier space: ∂/∂x_i → i*k_i, so ∂²/∂x_i∂x_j → -k_i*k_j
% Therefore: L[û] = Σᵢⱼ A(i,j) * (-k_i * k_j) * û

I = eye(N);
Elliptic_spec = zeros(N^d, N^d);

% For each coefficient A(i,j), build the operator for ∂²/∂x_i∂x_j
for i = 1:d
    for j = 1:d
        % Construct d-fold Kronecker product
        % Position i gets: D (contains factor of i*k_i)
        % Position j gets: D (contains factor of i*k_j)  
        % Other positions: I
        % Result: (i*k_i) * (i*k_j) = -k_i * k_j (the minus comes from i²)
        
        factors = cell(d, 1);
        for dim = 1:d
            if dim == i
                factors{dim} = D;  % i*k_i
            elseif dim == j && i ~= j
                factors{dim} = D;  % i*k_j (only if different dimension)
            else
                factors{dim} = I;
            end
        end
        
        % Compute Kronecker product of factors
        kron_result = factors{1};
        for dim = 2:d
            kron_result = kron(kron_result, factors{dim});
        end
        
        % Add to elliptic operator with coefficient
        % Note: (i*k_i)*(i*k_j) = -k_i*k_j is already in the Kronecker product
        Elliptic_spec = Elliptic_spec + A(i,j) * kron_result;
    end
end

fprintf('Elliptic operator built: %d×%d\n', size(Elliptic_spec, 1), size(Elliptic_spec, 2));

% Negate to match convention: L = -∇·(A∇u)
Elliptic_spec = -Elliptic_spec;

%% 6. HANDLE ZERO MODE AND CREATE INVERSE
% Avoid division by zero (DC component)
% Zero mode is the constant function, which has eigenvalue 0 for L
% We regularize by replacing it with 1

Elliptic_reg = Elliptic_spec;
Elliptic_reg(1, 1) = 1;  % Set (1,1) element to 1 to avoid singular matrix

diag_vals = diag(Elliptic_reg);
invElliptic = diag(1 ./ diag_vals);

fprintf('Conditioning: min|λ|=%.3e, max|λ|=%.3e\n', min(abs(diag_vals(2:end))), max(abs(diag_vals)));

%% 7. QUANTUM CIRCUIT CONSTRUCTION
fprintf('Building quantum circuit...\n');

total_qubits = d * n;
fprintf('Total qubits: %d\n', total_qubits);

% Fourier gates
FG = GroupFourier(d, n);
GF = FG.ctranspose();

% Diagonal gate with inverse elliptic operator
DiagEncoding = MakeUnitary(invElliptic);

% Circuit: iFFT -> Diagonal -> FFT
totalCircuit = qclab.QCircuit(total_qubits + 1);
totalCircuit.push_back(GF);  % iFFT
totalCircuit.push_back(qclab.qgates.MatrixGate(0:total_qubits, DiagEncoding, "Diagonal"));
totalCircuit.push_back(FG);  % FFT

% Extract matrix
totalMat_full = totalCircuit.matrix();
totalMat = totalMat_full(1:N^d, 1:N^d);

fprintf('Quantum operator: %d×%d\n', size(totalMat, 1), size(totalMat, 2));

%% 8. CREATE RHS FUNCTION AND SOLVE
% Define source function
if d == 1
    f_func = @(x1) cos(2*pi*x1);
elseif d == 2
    f_func = @(x1, x2) cos(2*pi*x1) .* sin(-2*pi*x2);
elseif d == 3
    f_func = @(x1, x2, x3) cos(2*pi*x1) .* sin(2*pi*x2) .* cos(2*pi*x3);
elseif d == 4
    f_func = @(x1, x2, x3, x4) cos(2*pi*x1) .* sin(2*pi*x2) .* cos(2*pi*x3) .* sin(2*pi*x4);
else
    % Generic: just use first dimension
    f_func = @(varargin) cos(2*pi*varargin{1});
    for k = 2:d
        f_func_old = f_func;
        if mod(k, 2) == 0
            f_func = @(varargin) f_func_old(varargin{:}) .* sin(2*pi*varargin{k});
        else
            f_func = @(varargin) f_func_old(varargin{:}) .* cos(2*pi*varargin{k});
        end
    end
end

% Evaluate RHS on grid
fprintf('Evaluating source term...\n');
f_grid = f_func(grids{:});
f_vec = f_grid(:);

fprintf('RHS vector size: %d\n', length(f_vec));

%% 9. SOLVE USING QUANTUM OPERATOR
fprintf('Solving via quantum operator...\n');

u_quantum_vec = totalMat * f_vec;
u_quantum = reshape(real(u_quantum_vec), grid_size);

fprintf('Solution computed.\n');

%% 10. CLASSICAL SOLUTION (Reference)
% Solve: L[u] = f using backslash (direct linear solve)
fprintf('Computing classical reference solution...\n');

% Solve the system: Elliptic_spec * u = f_vec
u_classical_vec = Elliptic_spec \ f_vec;
u_classical = reshape(real(u_classical_vec), grid_size);

%% 11. ERROR ANALYSIS
diff_norm = norm(u_classical - u_quantum, 'fro');
abs_err = abs(u_classical - u_quantum);
rel_err = abs_err ./ (abs(u_classical) + 1e-10);

fprintf('\n=== ERROR ANALYSIS ===\n');
fprintf('Frobenius norm: %.3e\n', diff_norm);
fprintf('Max absolute error: %.3e\n', max(abs_err(:)));
fprintf('Mean absolute error: %.3e\n', mean(abs_err(:)));
fprintf('Max relative error: %.3e\n', max(rel_err(:)));

%% 12. VISUALIZATION
fprintf('Creating visualizations...\n');

if d == 1
    % 1D: Line plots
    figure('Position', [50, 50, 1200, 600], 'Name', sprintf('%dD Spectral Elliptic Solver', d));
    
    subplot(1, 2, 1);
    plot(x, u_classical, 'b-', 'LineWidth', 2); hold on;
    plot(x, u_quantum, 'r--', 'LineWidth', 2);
    xlabel('x'); ylabel('u');
    title('Solution');
    legend('Classical', 'Quantum'); grid on;
    
    subplot(1, 2, 2);
    semilogy(x, abs_err, 'ko-', 'LineWidth', 1.5, 'MarkerSize', 3);
    xlabel('x'); ylabel('Error');
    title('Absolute Error');
    grid on;
    
elseif d == 2
    % 2D: Heatmaps
    figure('Position', [50, 50, 1400, 900], 'Name', sprintf('%dD Spectral Elliptic Solver', d));
    
    subplot(2, 3, 1);
    imagesc(x, x, u_classical');
    axis square; colorbar; title('Classical Solution');
    xlabel('x'); ylabel('y'); set(gca, 'YDir', 'normal');
    
    subplot(2, 3, 2);
    imagesc(x, x, u_quantum');
    axis square; colorbar; title('Quantum Solution');
    xlabel('x'); ylabel('y'); set(gca, 'YDir', 'normal');
    
    subplot(2, 3, 3);
    imagesc(x, x, abs_err');
    axis square; colorbar; title(sprintf('Absolute Error\nMax: %.2e', max(abs_err(:))));
    xlabel('x'); ylabel('y'); set(gca, 'YDir', 'normal');
    
    subplot(2, 3, 4);
    imagesc(x, x, rel_err');
    axis square; colorbar; title('Relative Error');
    xlabel('x'); ylabel('y'); set(gca, 'YDir', 'normal');
    
    mid = floor(N/2);
    subplot(2, 3, 5);
    plot(x, u_classical(:, mid), 'b-', 'LineWidth', 2); hold on;
    plot(x, u_quantum(:, mid), 'r--', 'LineWidth', 2);
    xlabel('x'); ylabel('u');
    title(sprintf('Cross-section at y=%.3f', x(mid)));
    legend('Classical', 'Quantum'); grid on;
    
    subplot(2, 3, 6);
    axis off;
    text(0.1, 0.8, 'Metrics', 'FontWeight', 'bold', 'FontSize', 12);
    text(0.1, 0.7, sprintf('Grid: %d×%d', N, N), 'FontSize', 10);
    text(0.1, 0.6, sprintf('Frobenius: %.3e', diff_norm), 'FontSize', 10);
    text(0.1, 0.5, sprintf('Max Error: %.3e', max(abs_err(:))), 'FontSize', 10);
    text(0.1, 0.4, sprintf('Mean Error: %.3e', mean(abs_err(:))), 'FontSize', 10);
    text(0.1, 0.3, sprintf('Qubits: %d', total_qubits), 'FontSize', 10);
    text(0.1, 0.2, sprintf('Grid pts: %d', N^d), 'FontSize', 10);
    box on;
    
elseif d == 3
    % 3D: Z-slices
    figure('Position', [50, 50, 1400, 900], 'Name', sprintf('%dD Spectral Elliptic Solver', d));
    
    z_slices = [1, floor(N/2), N];
    
    for slice_idx = 1:3
        z = z_slices(slice_idx);
        
        subplot(3, 3, slice_idx);
        imagesc(x, x, u_classical(:, :, z)');
        axis square; colorbar; title(sprintf('Classical z=%.3f', x(z)));
        set(gca, 'YDir', 'normal');
        
        subplot(3, 3, slice_idx + 3);
        imagesc(x, x, u_quantum(:, :, z)');
        axis square; colorbar; title(sprintf('Quantum z=%.3f', x(z)));
        set(gca, 'YDir', 'normal');
        
        subplot(3, 3, slice_idx + 6);
        slice_err = abs_err(:, :, z);
        imagesc(x, x, slice_err');
        axis square; colorbar; title(sprintf('Error z=%.3f', x(z)));
        set(gca, 'YDir', 'normal');
    end
    
else
    % High-D: 2D projection
    figure('Position', [50, 50, 1200, 900], 'Name', sprintf('%dD Spectral Elliptic Solver', d));
    
    mid_idx = repmat(floor(N/2), 1, d-2);
    
    if d == 4
        proj_cl = u_classical(:, :, mid_idx(1), mid_idx(2));
        proj_qu = u_quantum(:, :, mid_idx(1), mid_idx(2));
        proj_err = abs_err(:, :, mid_idx(1), mid_idx(2));
    elseif d == 5
        proj_cl = u_classical(:, :, mid_idx(1), mid_idx(2), mid_idx(3));
        proj_qu = u_quantum(:, :, mid_idx(1), mid_idx(2), mid_idx(3));
        proj_err = abs_err(:, :, mid_idx(1), mid_idx(2), mid_idx(3));
    else
        proj_cl = u_classical(:, :);
        proj_qu = u_quantum(:, :);
        proj_err = abs_err(:, :);
    end
    
    subplot(2, 3, 1);
    imagesc(x, x, proj_cl');
    axis square; colorbar; title('Classical (2D projection)');
    set(gca, 'YDir', 'normal');
    
    subplot(2, 3, 2);
    imagesc(x, x, proj_qu');
    axis square; colorbar; title('Quantum (2D projection)');
    set(gca, 'YDir', 'normal');
    
    subplot(2, 3, 3);
    imagesc(x, x, proj_err');
    axis square; colorbar; title(sprintf('Error (2D projection)\nMax: %.2e', max(proj_err(:))));
    set(gca, 'YDir', 'normal');
    
    subplot(2, 3, 4);
    axis off;
    text(0.1, 0.9, sprintf('Dimension: %d', d), 'FontSize', 12, 'FontWeight', 'bold');
    text(0.1, 0.8, sprintf('Grid: N=%d per dim', N), 'FontSize', 10);
    text(0.1, 0.7, sprintf('Total points: %d', N^d), 'FontSize', 10);
    text(0.1, 0.6, sprintf('Total qubits: %d', total_qubits), 'FontSize', 10);
    text(0.1, 0.5, '--- Errors ---', 'FontSize', 10, 'FontWeight', 'bold');
    text(0.1, 0.4, sprintf('Frobenius: %.3e', diff_norm), 'FontSize', 9);
    text(0.1, 0.3, sprintf('Max Error: %.3e', max(abs_err(:))), 'FontSize', 9);
    text(0.1, 0.2, sprintf('Mean Error: %.3e', mean(abs_err(:))), 'FontSize', 9);
    box on;
    
    subplot(2, 3, 5:6);
    axis off;
    text(0.1, 0.9, 'Spectral Elliptic Solver Notes:', 'FontSize', 11, 'FontWeight', 'bold');
    text(0.1, 0.8, sprintf('• Dimension: %dD', d), 'FontSize', 10);
    text(0.1, 0.7, sprintf('• Spectral derivative operator: D = diag(i·k)'), 'FontSize', 10);
    text(0.1, 0.6, sprintf('• Elliptic operator: L = Σᵢⱼ A(i,j)·Dᵢ²·Dⱼ²'), 'FontSize', 10);
    text(0.1, 0.5, sprintf('• Kronecker product for multi-dimensional operators'), 'FontSize', 10);
    text(0.1, 0.4, sprintf('• Quantum encoding via diagonal gate'), 'FontSize', 10);
    text(0.1, 0.3, sprintf('• Uses GroupFourier (qclab) for QFT'), 'FontSize', 10);
    box on;
end

fprintf('Visualization complete.\n');

%% 13. SUMMARY
fprintf('\n=== SUMMARY ===\n');
fprintf('Problem: Solve L[u] = f\n');
fprintf('Dimension: %d\n', d);
fprintf('Grid points per dimension: %d (N=2^%d)\n', N, n);
fprintf('Total grid points: %d\n', N^d);
fprintf('Matrix A:\n');
disp(A);
fprintf('\nQuantum circuit: %d qubits\n', total_qubits);
fprintf('\nError metrics:\n');
fprintf('  Frobenius norm: %.3e\n', diff_norm);
fprintf('  Max absolute error: %.3e\n', max(abs_err(:)));
fprintf('  Mean absolute error: %.3e\n', mean(abs_err(:)));

fprintf('\n✓ Solver complete!\n');