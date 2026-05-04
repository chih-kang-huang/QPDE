%% Quantum Heat Equation Solver (Corrected Order)
clear all; close all; clc;

%% ========== SETUP & PARAMETERS ==========
% Physical Parameters matching Python script
x_lb = 0;
x_rb = 1;
d = 2;
n = 5;          % N = 32
dt = 1e-3;
N = 2^n;
L_dom = x_rb - x_lb;
dx = L_dom / N;

% Grid Construction
x = linspace(x_lb, x_rb, N+1); x(end) = []; 
[xs, ys] = ndgrid(x, x);

%% ========== DATA INPUT ==========
f_source = @(x, y) cos(2*pi*x) .* sin(-2*pi*y);

u_init_func = @(x, y) cos(2*pi*x) .* sin(8*pi*y) + ...
                      2 * sin(6*pi*y) + ...
                      3 * sin(10*pi*x) .* cos(12*pi*y).^2;

%% ========== SPECTRAL OPERATOR CONSTRUCTION ==========
function [QuantumOp, alpha, inv_diff_op] = build_heat_operator(N, n_qubits, dt, dx)
    
    % 1. Define Spectral Frequencies
    k_vec = (2*pi) * fftfreq_vec(N, 1);
    [kx, ky] = ndgrid(k_vec, k_vec);
    
    % 2. Laplacian Eigenvalues: -k^2
    laplacian_eigenvalues = -(kx.^2 + ky.^2);
    
    % 3. Construct Semi-Implicit Operator: (1 - dt*Laplacian)^-1
    denom = 1 - dt * laplacian_eigenvalues;
    inv_diff_op = 1 ./ denom;
    
    % 4. Normalization for Block Encoding
    alpha = max(abs(inv_diff_op(:)));
    normalized_diag = inv_diff_op(:) / alpha;
    
    % 5. Build Quantum Circuit
    fprintf('Building Quantum Operator...\n');
    fprintf('   - Normalization factor (alpha): %.4f\n', alpha);
    
    FG = GroupFourier(2, n_qubits); % 2D QFT
    GF = FG.ctranspose();           % Inverse QFT
    
    DiagEncoding = MakeUnitary(diag(normalized_diag));
    
    totalCircuit = qclab.QCircuit(2*n_qubits + 1);
    
    % --- CORRECTED SEQUENCE ---
    % Execution Order: QFT -> Diagonal -> iQFT
    % Mathematical Order: M = iQFT * Diag * QFT
    totalCircuit.push_back(FG); 
    totalCircuit.push_back(qclab.qgates.MatrixGate(0:2*n_qubits, DiagEncoding, "Diagonal"));
    totalCircuit.push_back(GF);
    
    % Extract Matrix
    totalMat = totalCircuit.matrix;
    dim = 2^(2*n_qubits);
    QuantumOp = totalMat(1:dim, 1:dim);
    
end

%% ========== SOLVERS ==========
% Classical Reference Solver (Spectral)
function u_next = classical_heat_step(u_curr, f_val, inv_diff_op, dt)
    u_h = fft2(u_curr);
    f_h = fft2(f_val);
    
    rhs_h = u_h - dt * f_h;
    u_next_h = rhs_h .* inv_diff_op;
    
    u_next = real(ifft2(u_next_h));
end

% Quantum Solver
function u_next = quantum_heat_step(u_curr, f_val, QuantumOp, alpha, dt, N)
    % 1. Form RHS vector: v = u - dt*f
    v = u_curr - dt * f_val;
    v_vec = v(:);
    
    % 2. Apply Quantum Operator
    % Result = Op * v * alpha
    u_next_vec = (QuantumOp * v_vec) * alpha;
    
    u_next = reshape(real(u_next_vec), N, N);
end

%% ========== MAIN EXECUTION ==========

% 1. Initialize
u_quantum = u_init_func(xs, ys);
u_classical = u_quantum;
f_val = f_source(xs, ys);

% 2. Build Operators
[Q_Op, alpha, filter_op] = build_heat_operator(N, n, dt, dx);

% 3. Histories
calc_energy = @(u, kx, ky) 0.5 * mean( real(ifft2(1i*kx.*fft2(u))).^2 + ...
                                       real(ifft2(1i*ky.*fft2(u))).^2, 'all');

k_vec_e = (2*pi) * fftfreq_vec(N, 1);
[kx_e, ky_e] = ndgrid(k_vec_e, k_vec_e);

E0 = calc_energy(u_classical, kx_e, ky_e);
energy_classical_hist = [E0];
energy_quantum_hist = [E0];

% 4. Time Loop
num_steps = 40;
fprintf('Starting Time Evolution (%d steps)...\n', num_steps);

for t = 1:num_steps
    u_classical = classical_heat_step(u_classical, f_val, filter_op, dt);
    u_quantum = quantum_heat_step(u_quantum, f_val, Q_Op, alpha, dt, N);
    
    energy_classical_hist(end+1) = calc_energy(u_classical, kx_e, ky_e);
    energy_quantum_hist(end+1) = calc_energy(u_quantum, kx_e, ky_e);
    
    if mod(t, 10) == 0
        fprintf('Step %d completed.\n', t);
    end
end

%% ========== VISUALIZATION ==========
figure('Position', [100 100 1600 500], 'Name', 'Heat Equation Results');

subplot(1, 4, 1);
plot(0:num_steps, energy_classical_hist, 'b-', 'LineWidth', 2); hold on;
plot(0:num_steps, energy_quantum_hist, 'r--', 'LineWidth', 2);
title('Energy Decay'); legend('Classical', 'Quantum'); grid on;

subplot(1, 4, 2);
imagesc([x_lb, x_rb], [x_lb, x_rb], u_quantum');
axis square; colorbar; title('Quantum Solution'); set(gca, 'YDir', 'normal');

subplot(1, 4, 3);
imagesc([x_lb, x_rb], [x_lb, x_rb], u_classical');
axis square; colorbar; title('Classical Solution'); set(gca, 'YDir', 'normal');

subplot(1, 4, 4);
err = abs(u_classical - u_quantum);
imagesc([x_lb, x_rb], [x_lb, x_rb], err');
axis square; colorbar;
title(sprintf('Abs Error\nMean: %.2e', mean(err(:)))); set(gca, 'YDir', 'normal');

%% ========== HELPER FUNCTIONS ==========
function freq = fftfreq_vec(N, ~)
    if mod(N, 2) == 0
        freq = [0:N/2-1, -N/2:-1]';
    else
        freq = [0:(N-1)/2, -(N-1)/2:-1]';
    end
end