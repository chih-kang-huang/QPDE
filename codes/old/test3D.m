%% Spectral method setup for 3D
clear; clc; close all;

% Parameters
n = 2;              % Qubits per dimension (N = 16)
N = 2^n;
dt = 1e-3;
steps = 50;

% Domain setup
x_lb = 0;
x_rb = 1;
L = x_rb - x_lb;
dx = L / N;
x = x_lb + (0:N-1) * dx;

% 3D grid
[xs, ys, zs] = ndgrid(x, x, x);

% Elliptic operator coefficient matrix
A = [3.0  1.0  0.0;
     1.0  2.0  0.0;
     0.0  0.0  3.0];

% Fourier transforms
FG = GroupFourier(3, n);
GF = FG.ctranspose();

%% Build spectral operators

% Spectral eigenvalues for 1D derivative operator
D = diag(spectral_eigenvalues(N));
I = eye(N);

% Elliptic operator in spectral space using Kronecker products
Elliptic_spec = ...
    A(1,1) * kron(kron(D^2, I), I) + ...
    A(2,2) * kron(kron(I, D^2), I) + ...
    A(3,3) * kron(kron(I, I), D^2) + ...
    2*A(1,2) * kron(kron(D, D), I) + ...
    2*A(2,3) * kron(kron(I, D), D) + ...
    2*A(1,3) * kron(kron(D, I), D);

% Diagonal filter for energy computation
k_ind = [0:N/2-1, -N/2:-1]';
k_base = 2i * pi * k_ind / (N * dx);
[Kx, Ky, Kz] = ndgrid(k_base, k_base, k_base);

Elliptic_diag = ...
    A(1,1) * (Kx.^2) + A(2,2) * (Ky.^2) + A(3,3) * (Kz.^2) + ...
    2*A(1,2) * (Kx .* Ky) + 2*A(2,3) * (Ky .* Kz) + 2*A(1,3) * (Kz .* Kx);

% Precompute filter: (I - dt*Elliptic)^{-1}
Filter_spec = ((eye(N^3) - dt * Elliptic_spec) \ eye(N^3));

%% Quantum circuit and operator

% Unitary encoding of filter
DiagEncoding = MakeUnitary(Filter_spec);

% Build circuit: iQFT -> Diagonal Gate -> QFT
totalCircuit = qclab.QCircuit(3*n + 1);
totalCircuit.push_back(GF);
totalCircuit.push_back(qclab.qgates.MatrixGate(0:3*n, DiagEncoding, "Diagonal"));
totalCircuit.push_back(FG);

% Extract quantum operator
totalMat = totalCircuit.matrix;
Q_Op = totalMat(1:N^3, 1:N^3);

%% Define source term and initial condition
f = @(x, y, z) cos(2*pi*x) .* sin(-2*pi*y) .* cos(2*pi*z);

u_init = @(x, y, z) cos(2*pi*x) .* sin(8*pi*y) + ...
                     2*sin(6*pi*y) + ...
                     3*sin(10*pi*x) .* cos(12*pi*y).^2 + ...
                     sin(2*pi*z);

%% Time evolution

% Initialize solutions
u_class = u_init(xs, ys, zs);
u_quant = u_init(xs, ys, zs);

% Energy history
energy_hist_class = zeros(1, steps+1);
energy_hist_quant = zeros(1, steps+1);

% Energy computation
calc_energy = @(u_in) 0.5 * real(sum(sum(sum(conj(fftn(u_in)) .* (-Elliptic_diag) .* fftn(u_in))))) / N^(3*log2(N));

energy_hist_class(1) = calc_energy(u_class);
energy_hist_quant(1) = calc_energy(u_quant);

fprintf('Running %d diffusion steps (N=%d)...\n', steps, N);

% Prepare quantum operator
alpha = max(abs(diag(Filter_spec)));
DiagEncoding_evo = MakeUnitary(Filter_spec / alpha);
circuit_evo = qclab.QCircuit(3*n + 1);
circuit_evo.push_back(GF);
circuit_evo.push_back(qclab.qgates.MatrixGate(0:3*n, DiagEncoding_evo, "Diagonal"));
circuit_evo.push_back(FG);
Q_Op_evo = circuit_evo.matrix;
Q_Op_evo=Q_Op_evo(1:N^3, 1:N^3);
% Time stepping loop
for t = 1:steps
    f_vals = f(xs, ys, zs);
    
    % Classical
    v_class = u_class - dt * f_vals;
    u_class = solver_Elliptic_3D(v_class, Filter_spec, N);
    
    % Quantum
    v_quant = u_quant - dt * f_vals;
    u_quant = reshape(real((Q_Op_evo * v_quant(:)) * alpha), N, N, N);
    
    % Energy
    energy_hist_class(t+1) = calc_energy(u_class);
    energy_hist_quant(t+1) = calc_energy(u_quant);
    
    fprintf('Step %d: E_cl = %.6f, E_qu = %.6f\n', t, energy_hist_class(t+1), energy_hist_quant(t+1));
end

%% Error metrics

u_class_vec = u_class(:);
u_quant_vec = u_quant(:);

error_norm_frobenius = norm(u_class_vec - u_quant_vec, 'fro');
error_norm_l2 = norm(u_class_vec - u_quant_vec, 2);
energy_err = abs(energy_hist_class - energy_hist_quant);

fprintf('\n=== Final Results ===\n');
fprintf('Frobenius Norm: %e\n', error_norm_frobenius);
fprintf('L2 Norm: %e\n', error_norm_l2);
fprintf('Max Error: %e\n', max(abs(u_class_vec - u_quant_vec)));
fprintf('Mean Error: %e\n', mean(abs(u_class_vec - u_quant_vec)));
fprintf('Final E_class: %.6f\n', energy_hist_class(end));
fprintf('Final E_quant: %.6f\n', energy_hist_quant(end));

%% Visualization

figure('Position', [50, 50, 1600, 800], 'Name', '3D Diffusion Evolution');

% Energy evolution
subplot(2, 4, 1);
plot(0:steps, energy_hist_class, 'b-', 'LineWidth', 2); hold on;
plot(0:steps, energy_hist_quant, 'r--', 'LineWidth', 2);
title('Energy Evolution'); xlabel('Time Step'); ylabel('Energy');
legend('Classical', 'Quantum'); grid on;

% Energy error
subplot(2, 4, 2);
semilogy(0:steps, energy_err, 'ko-', 'LineWidth', 1.5, 'MarkerSize', 4);
title('Energy Error'); xlabel('Time Step'); ylabel('|E_cl - E_qu|'); grid on;

% Classical solution
subplot(2, 4, 3);
imagesc(xs(:,:,1), ys(:,:,1), real(u_class(:,:,1))');
axis image; colorbar; title('Classical (z=0)');
xlabel('x'); ylabel('y'); set(gca, 'YDir', 'normal');

% Quantum solution
subplot(2, 4, 4);
imagesc(xs(:,:,1), ys(:,:,1), real(u_quant(:,:,1))');
axis image; colorbar; title('Quantum (z=0)');
xlabel('x'); ylabel('y'); set(gca, 'YDir', 'normal');

% Absolute error
abs_err = abs(u_class(:,:,1) - u_quant(:,:,1));
subplot(2, 4, 5);
imagesc(xs(:,:,1), ys(:,:,1), abs_err');
axis image; colorbar; title(sprintf('Abs Error (max: %.2e)', max(abs_err(:))));
xlabel('x'); ylabel('y'); set(gca, 'YDir', 'normal');

% Relative error
rel_err = abs_err ./ (abs(u_class(:,:,1)) + 1e-10);
subplot(2, 4, 6);
imagesc(xs(:,:,1), ys(:,:,1), rel_err');
axis image; colorbar; title('Relative Error');
xlabel('x'); ylabel('y'); set(gca, 'YDir', 'normal');

% Cross-section
subplot(2, 4, 7);
mid = floor(N/2);
plot(x, u_class(:, mid, 1), 'b-', 'LineWidth', 2); hold on;
plot(x, u_quant(:, mid, 1), 'r--', 'LineWidth', 2);
title(sprintf('Cross-section (y=%.2f, z=0)', x(mid)));
legend('Classical', 'Quantum'); grid on; xlabel('x');

% Metrics
subplot(2, 4, 8);
axis off;
txt = sprintf(['Steps: %d, N: %d\n', ...
               'Frob: %.2e\n', ...
               'L2: %.2e\n', ...
               'Max: %.2e\n', ...
               'Mean: %.2e\n', ...
               'E_cl: %.4f\n', ...
               'E_qu: %.4f\n', ...
               'dt: %.1e'], ...
              steps, N, error_norm_frobenius, error_norm_l2, ...
              max(abs(u_class_vec - u_quant_vec)), mean(abs(u_class_vec - u_quant_vec)), ...
              energy_hist_class(end), energy_hist_quant(end), dt);
text(0.1, 0.5, txt, 'FontSize', 10, 'VerticalAlignment', 'middle');
box on;

%% Helper Functions

function k = spectral_eigenvalues(N, L)
    % 1D derivative operator eigenvalues with periodic boundary conditions
    if nargin < 2, L = 1.0; end
    freqs = [0:(N/2-1), -(N/2):-1] / N;
    k = 2i * pi * freqs;
end


function u = solver_Elliptic_3D(f_or_u, Filter_spec, N)
    % Solve (I - dt*div(A*grad)) u = f using precomputed filter
    % Input: f_or_u (function handle or matrix), Filter_spec, grid size N
    
    if isa(f_or_u, 'function_handle')
        f_vals = f_or_u;
    else
        f_vals = f_or_u;
    end
    
    f_flatten = f_vals(:);
    u_flatten = Filter_spec * f_flatten;
    u = reshape(real(u_flatten), N, N, N);
end