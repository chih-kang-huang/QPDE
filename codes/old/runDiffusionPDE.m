%% Main Script: Run Quantum Diffusion Simulation
clear; clc; close all;

%% 1. Setup Parameters
n = 5;              % Qubits (N=32)
N = 2^n;
dt = 1e-3;
steps = 40;
L = 1;
dx = L/N;
x = (0:N-1)*dx;
[X, Y] = ndgrid(x, x);

% Diffusion Tensor (Generic Square A)
A = [3.0, 1.0; 
     1.0, 2.0];

% Initial Conditions & Source
f = cos(2*pi*X) .* sin(-2*pi*Y);
u_init = cos(2*pi*X) .* sin(8*pi*Y) + 2*sin(6*pi*Y) + 3*sin(10*X).*cos(12*Y).^2;

%% 2. Run Quantum Solver (Using Function)
fprintf('Running Quantum Solver...\n');
tic;
[u_quant, hist_quant] = QuantumDiffusionSolver(A, N, dt, steps, u_init, f);
t_quant = toc;
fprintf('Quantum solver finished in %.2f s.\n', t_quant);

%% 3. Run Classical Reference (for Validation)
fprintf('Running Classical Reference...\n');
% Re-implementing classical spectral step locally for comparison
k_vec = 2i*pi*[0:N/2-1, -N/2:-1]';
[KX, KY] = ndgrid(k_vec, k_vec);
Elliptic = A(1,1)*KX.^2 + (A(1,2)+A(2,1))*KX.*KY + A(2,2)*KY.^2;
Filter = 1 ./ (1 - dt * Elliptic);

u_class = u_init;
hist_class.energy = zeros(1, steps+1);
hist_class.time = hist_quant.time;
calc_E = @(u) 0.5 * real(sum(sum( conj(fft2(u)) .* (-Elliptic) .* fft2(u) ))) / N^4;
hist_class.energy(1) = calc_E(u_class);

for t = 1:steps
    u_h = fft2(u_class);
    f_h = fft2(f);
    u_class = real(ifft2( (u_h - dt*f_h) .* Filter ));
    hist_class.energy(t+1) = calc_E(u_class);
end

%% 4. Calculate Metrics
abs_err = abs(u_class - u_quant);
rel_err = abs_err ./ (abs(u_class) + 1e-10);
diff_norm = norm(u_class(:) - u_quant(:), 'fro');
energy_err = abs(hist_class.energy - hist_quant.energy);

%% 5. Visualization (2x4 Grid)
figure('Position', [50, 50, 1600, 800], 'Name', 'Quantum Diffusion Results');

% (2,4,1) Energy Evolution
subplot(2, 4, 1);
plot(hist_class.time, hist_class.energy, 'b-', 'LineWidth', 2); hold on;
plot(hist_quant.time, hist_quant.energy, 'r--', 'LineWidth', 2);
title('Energy Evolution'); xlabel('t'); ylabel('Energy');
legend('Classical', 'Quantum'); grid on;

% (2,4,2) Absolute Energy Error
subplot(2, 4, 2);
semilogy(hist_class.time, energy_err, 'k-o', 'LineWidth', 1.5, 'MarkerSize', 4);
title('Abs. Energy Error'); xlabel('t'); ylabel('|E_{cl} - E_{qu}|'); grid on;

% (2,4,3) Classical Solution
subplot(2, 4, 3);
imagesc(u_class'); axis square; colorbar;
title('Classical Solution'); xlabel('x'); ylabel('y'); set(gca,'YDir','normal');

% (2,4,4) Quantum Solution
subplot(2, 4, 4);
imagesc(u_quant'); axis square; colorbar;
title('Quantum Solution'); xlabel('x'); ylabel('y'); set(gca,'YDir','normal');

% (2,4,5) Absolute Error
subplot(2, 4, 5);
imagesc(abs_err'); axis square; colorbar;
title('Absolute Error'); xlabel('x'); ylabel('y'); set(gca,'YDir','normal');

% (2,4,6) Relative Error
subplot(2, 4, 6);
imagesc(rel_err'); axis square; colorbar;
title('Relative Error'); xlabel('x'); ylabel('y'); set(gca,'YDir','normal');

% (2,4,7) Cross-section
subplot(2, 4, 7);
mid = floor(N/2);
plot(x, u_class(:, mid), 'b-', 'LineWidth', 2); hold on;
plot(x, u_quant(:, mid), 'r--', 'LineWidth', 2);
title(['Cross-section y=' num2str(x(mid))]); 
legend('Classical', 'Quantum'); grid on;

% (2,4,8) Metrics
subplot(2, 4, 8); axis off;
text(0.05, 0.9, 'Simulation Metrics', 'FontWeight', 'bold', 'FontSize', 12);
text(0.05, 0.7, sprintf('Grid: %dx%d (N=%d)', N, N, N));
text(0.05, 0.6, sprintf('Steps: %d, dt: %.1e', steps, dt));
text(0.05, 0.5, sprintf('Frobenius Error: %.3e', diff_norm));
text(0.05, 0.4, sprintf('Max Abs Error: %.3e', max(abs_err(:))));
text(0.05, 0.3, sprintf('Alpha (Norm): %.4f', hist_quant.alpha));
box on;