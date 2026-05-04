%% RunAllQPDE_Tests.m
clear; clc; close all;
warning('off', 'all')
%% =========================================================
%  GLOBAL PARAMETERS
% ==========================================================
dt    = 1e-3;   % Time step (diffusion solver only)
steps = 100;    % Number of time steps (diffusion solver only)

diffusion  = true;
elliptic   = false;
helmholtz  = false;

k_range    = 1:5;

%% =========================================================
%  CONFIGURATION GRID
% ==========================================================
dims_to_test = [2, 3];
ns_per_dim = containers.Map([2, 3], {[4], [3]});
% ns_per_dim = containers.Map([2, 3], {[2], [1]});

% ---- A MATRIX CONFIGURATIONS --------------------------------
A_configs = { ...
    struct('label', 'Identity', 'fun', @(d) eye(d), 'dims', [])
};

% HARDCODED MATRICES (add new ones here — set 'dims' to restrict dimension)
A_configs{end+1} = struct('label', 'Custom', 'fun', @(d) [3,1;1,2],         'dims', [2]);
A_configs{end+1} = struct('label', 'Custom', 'fun', @(d) [10,0;0,1],        'dims', [2]);
A_configs{end+1} = struct('label', 'Custom', 'fun', @(d) [100,0;0,1],       'dims', [2]);
A_configs{end+1} = struct('label', 'Custom', 'fun', @(d) [100,0;0,0.1],     'dims', [2]);
A_configs{end+1} = struct('label', 'Custom', 'fun', @(d) [100000,0;0,1],    'dims', [2]);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A_configs{end+1} = struct('label', 'Custom', 'fun', @(d) [3,1,0.5;1,3,1;0.5,1,3],   'dims', [3]);
% A_configs{end+1} = struct('label', 'Custom', 'fun', @(d) [10,0,0;0,1,0;0,0,1],       'dims', [3]);
% A_configs{end+1} = struct('label', 'Custom', 'fun', @(d) [1,0,0;0,100,0;0,0,1],      'dims', [3]);
% A_configs{end+1} = struct('label', 'Custom', 'fun', @(d) [1,0,0;0,100,0;0,0,0.1],    'dims', [3]);
% A_configs{end+1} = struct('label', 'Custom', 'fun', @(d) [1,0,0;0,1,0;0,0,100000],   'dims', [3]);

%% =========================================================
%  COUNT VALID CONFIGURATIONS (respecting dims filter)
% ==========================================================
n_valid_A_diff     = 0;
n_valid_A_ell_helm = 0;
for iDim = 1:numel(dims_to_test)
    dim        = dims_to_test(iDim);
    ns_to_test = ns_per_dim(dim);
    for iA = 1:numel(A_configs)
        d_filter = A_configs{iA}.dims;
        if isempty(d_filter) || ismember(dim, d_filter)
            n_valid_A_diff     = n_valid_A_diff     + numel(ns_to_test);
            n_valid_A_ell_helm = n_valid_A_ell_helm + numel(ns_to_test);
        end
    end
end

% Helmholtz: dim=2 only
n_valid_dimN_helm = numel(ns_per_dim(2)) * numel(k_range);

n_total = n_valid_A_diff     *  diffusion                    + ...
          n_valid_A_ell_helm *  elliptic                     + ...
          n_valid_dimN_helm  * (helmholtz );

fprintf('======================================================\n');
fprintf('  QPDE Test Suite  |  %d total configurations\n', n_total);
fprintf('======================================================\n\n');

%% =========================================================
%  RUN ALL TESTS
% ==========================================================
test_id = 0;
results = struct();

for iDim = 1:numel(dims_to_test)
    dim        = dims_to_test(iDim);
    ns_to_test = ns_per_dim(dim);   

    fDiff       = make_f_diff(dim);
    uInit       = make_u_init(dim);
    fAnalytical = make_f_analytical_diff(dim);
    fEll        = make_f_ell(dim);
    uTrueEll    = make_u_true_ell(dim);

    %% ---- (1) DIFFUSION SOLVER -----------------------------------
    if diffusion
        for iN = 1:numel(ns_to_test)
            n = ns_to_test(iN);
            N = 2^n;

            for iA = 1:numel(A_configs)
                d_filter = A_configs{iA}.dims;
                if ~isempty(d_filter) && ~ismember(dim, d_filter); continue; end

                A      = A_configs{iA}.fun(dim);
                Alabel = A_configs{iA}.label;

                is_symmetric = norm(A - A', 'fro') < 1e-10;
                is_spd       = all(eig(A) > 0);
                if ~is_symmetric || ~is_spd
                    fprintf('  SKIP: A="%s" is not SPD.\n\n', Alabel);
                    continue
                end

                test_id = test_id + 1;
                tag = sprintf('[%02d/%02d | Diffusion | dim=%d | N=2^%d=%d | A=%s]', ...
                              test_id, n_total, dim, n, N, Alabel);
                fprintf('%s\n', tag);
                fprintf('  A =\n'); disp(A);
                try
                    GenericDiffusion_QPDE(fDiff, uInit, A, N, dim, dt, steps, fAnalytical);
                    results(test_id).status = 'PASS';
                catch ME
                    results(test_id).status = 'FAIL';
                    results(test_id).error  = ME.message;
                    fprintf('  !! ERROR: %s\n', ME.message);
                end
                results(test_id).tag    = tag;
                results(test_id).solver = 'Diffusion';
                results(test_id).dim    = dim;
                results(test_id).N      = N;
                results(test_id).Alabel = Alabel;
                results(test_id).k      = NaN;
                % fprintf('  Status: %s\n\n', results(test_id).status);
            end % iA
        end % iN
    end

    %% ---- (2) ELLIPTIC & (3) HELMHOLTZ ---------------------------
    for iN = 1:numel(ns_to_test)
        n = ns_to_test(iN);
        N = 2^n;

        %% ---- (2) ELLIPTIC SOLVER --------------------------------
        if elliptic
            for iA = 1:numel(A_configs)
                d_filter = A_configs{iA}.dims;
                if ~isempty(d_filter) && ~ismember(dim, d_filter); continue; end

                A      = A_configs{iA}.fun(dim);
                Alabel = A_configs{iA}.label;

                is_symmetric = norm(A - A', 'fro') < 1e-10;
                is_spd       = all(eig(A) > 0);
                if ~is_symmetric || ~is_spd
                    fprintf('  SKIP: A="%s" is not SPD.\n\n', Alabel);
                    continue
                end

                test_id = test_id + 1;
                tag = sprintf('[%02d/%02d | Elliptic  | dim=%d | N=2^%d=%d | A=%s]', ...
                              test_id, n_total, dim, n, N, Alabel);
                fprintf('%s\n', tag);
                fprintf('  A =\n'); disp(A);
                try
                    GenericElliptic_QPDE(fEll, A, N, dim, uTrueEll);
                    results(test_id).status = 'PASS';
                catch ME
                    results(test_id).status = 'FAIL';
                    results(test_id).error  = ME.message;
                    fprintf('  !! ERROR: %s\n', ME.message);
                end
                results(test_id).tag    = tag;
                results(test_id).solver = 'Elliptic';
                results(test_id).dim    = dim;
                results(test_id).N      = N;
                results(test_id).Alabel = Alabel;
                results(test_id).k      = NaN;
                fprintf('  Status: %s\n\n', results(test_id).status);
            end % iA
        end

        %% ---- (3) HELMHOLTZ — dim=2 only, no A loop --------------
        if helmholtz && dim == 2
            for k_val = k_range
                k_sq = k_val^2;
                test_id = test_id + 1;
                tag = sprintf('[%02d/%02d | Helmholtz | dim=%d | N=2^%d=%d | k=%d (k^2=%d)]', ...
                              test_id, n_total, dim, n, N, k_val, k_sq);
                fprintf('%s\n', tag);
                try
                    GenericHelmoltz_QPDE(fEll, k_sq, N, dim, uTrueEll);
                    results(test_id).status = 'PASS';
                catch ME
                    results(test_id).status = 'FAIL';
                    results(test_id).error  = ME.message;
                    fprintf('  !! ERROR: %s\n', ME.message);
                end
                results(test_id).tag    = tag;
                results(test_id).solver = 'Helmholtz';
                results(test_id).dim    = dim;
                results(test_id).N      = N;
                results(test_id).Alabel = 'N/A';
                results(test_id).k      = k_val;
                %fprintf('  Status: %s\n\n', results(test_id).status);
            end % k_val
        end

    end % iN ell/helm
end % iDim

%% =========================================================
%  SUMMARY TABLE
% ==========================================================
fprintf('======================================================\n');
fprintf('  SUMMARY\n');
fprintf('======================================================\n');
fprintf('%-5s | %-10s | %-5s | %-6s | %-14s | %-8s | %s\n', ...
        'Test', 'Solver', 'dim', 'N', 'A', 'k (k^2)', 'Status');
fprintf('%s\n', repmat('-', 1, 72));

n_pass = 0; n_fail = 0;
for k = 1:numel(results)
    pass = strcmp(results(k).status, 'PASS');
    if pass; n_pass = n_pass + 1; else; n_fail = n_fail + 1; end

    if isnan(results(k).k)
        k_str = 'N/A';
    else
        k_str = sprintf('%d (%d)', results(k).k, results(k).k^2);
    end

    fprintf('%-5d | %-10s | %-5d | %-6d | %-14s | %-8s | %s\n', ...
            k, results(k).solver, results(k).dim, results(k).N, ...
            results(k).Alabel, k_str, results(k).status);
    if ~pass
        fprintf('  Error: %s\n', results(k).error);
    end
end

fprintf('%s\n', repmat('-', 1, 72));
fprintf('TOTAL: %d/%d passed  |  %d failed\n', n_pass, n_pass+n_fail, n_fail);
fprintf('======================================================\n');