%% RunAllQPDE_Tests.m

clear; clc; close all;
warning('off', 'all')
%% =========================================================
%  GLOBAL PARAMETERS
% ==========================================================
dt    = 1e-3;   % Time step (diffusion solver only)
steps = 100;    % Number of time steps (diffusion solver only)

%% =========================================================
%  CONFIGURATION GRID  
% ==========================================================
dims_to_test = [2];      
ns_to_test   = [3];      

% ---- A MATRIX CONFIGURATIONS --------------------------------
% Each entry has:
%   label : display name
%   fun   : @(d) -> A  factory function
%   dims  : row vector of valid dimensions ([] = all dims)
%
% IDENTITY / GENERATED MATRICES (work for any dimension)
A_configs = { ...
    struct('label', 'Identity',      'fun', @(d) eye(d),'dims', [])
};

% HARDCODED MATRICES (add new ones here — set 'dims' to restrict dimension)
A_configs{end+1} = struct('label', 'Custom',    'fun', @(d) [3,1;1,2], 'dims', [2]);

%% =========================================================
%  FUNCTION HANDLE FACTORY
%  Returns handles with the correct arity for the given dim.
% ==========================================================

function h = make_f_diff(dim)
    switch dim
        case 2; h = @(x,y)   cos(2*pi*x) .* sin(-4*pi*y);
        case 3; h = @(x,y,z) cos(2*pi*x) .* sin(-4*pi*y) .* cos(2*pi*z);
    end
end

function h = make_u_init(dim)
    switch dim
        case 2; h = @(x,y)   cos(2*pi*x).*sin(8*pi*y) + ...
                              2*sin(6*pi*y) + ...
                              3*sin(10*pi*x).*(cos(12*pi*y).^2);
        case 3; h = @(x,y,z) cos(2*pi*x).*sin(8*pi*y) + ...
                              2*sin(6*pi*y) + ...
                              3*sin(10*pi*x).*(cos(12*pi*y).^2) + ...
                              cos(2*pi*z);
    end
end

function h = make_f_analytical_diff(dim)
    switch dim
        case 2; h = @(x,y)   cos(2*pi*x) .* sin(-4*pi*y);
        case 3; h = @(x,y,z) cos(2*pi*x) .* sin(-4*pi*y) .* cos(2*pi*z);
    end
end

function h = make_f_ell(dim)
    switch dim
        case 2; h = @(x,y)   cos(2*pi*x) .* sin(-4*pi*y);
        case 3; h = @(x,y,z) 5*sin(2*pi*x) .* sin(2*pi*y) .* sin(2*pi*z);
    end
end

function h = make_u_true_ell(dim)
    switch dim
        case 2; h = @(x,y)  -cos(2*pi*x) .* sin(-4*pi*y) / (20*pi^2);
        case 3; h = @(x,y,z)-5*sin(2*pi*x).*sin(2*pi*y).*sin(2*pi*z) / (12*pi^2);
    end
end

%% =========================================================
%  COUNT VALID CONFIGURATIONS (respecting dims filter)
% ==========================================================
n_valid = 0;
for iDim = 1:numel(dims_to_test)
    for iN = 1:numel(ns_to_test)
        for iA = 1:numel(A_configs)
            d_filter = A_configs{iA}.dims;
            if isempty(d_filter) || ismember(dims_to_test(iDim), d_filter)
                n_valid = n_valid + 1;
            end
        end
    end
end
n_total = 2 * n_valid;   % diffusion + elliptic per config

fprintf('======================================================\n');
fprintf('  QPDE Test Suite  |  %d total configurations\n', n_total);
fprintf('======================================================\n\n');

%% =========================================================
%  RUN ALL TESTS
% ==========================================================
test_id = 0;
results = struct();

for iDim = 1:numel(dims_to_test)
    dim = dims_to_test(iDim);

    fDiff       = make_f_diff(dim);
    uInit       = make_u_init(dim);
    fAnalytical = make_f_analytical_diff(dim);
    fEll        = make_f_ell(dim);
    uTrueEll    = make_u_true_ell(dim);

    for iN = 1:numel(ns_to_test)
        n = ns_to_test(iN);
        N = 2^n;

        for iA = 1:numel(A_configs)

            % --- Skip if this matrix is not valid for current dim ---
            d_filter = A_configs{iA}.dims;
            if ~isempty(d_filter) && ~ismember(dim, d_filter)
                continue
            end

            A      = A_configs{iA}.fun(dim);
            Alabel = A_configs{iA}.label;

            % --- Validate A before running ---
            is_symmetric = norm(A - A', 'fro') < 1e-10;
            is_spd       = all(eig(A) > 0);
            if ~is_symmetric || ~is_spd
                fprintf('  SKIP: A="%s" is not symmetric positive definite.\n\n', Alabel);
                continue
            end

            %% ---- (1) DIFFUSION SOLVER --------------------------------
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
            fprintf('  Status: %s\n\n', results(test_id).status);

            %% ---- (2) ELLIPTIC SOLVER ---------------------------------
            test_id = test_id + 1;
            tag = sprintf('[%02d/%02d | Elliptic  | dim=%d | N=2^%d=%d | A=%s]', ...
                          test_id, n_total, dim, n, N, Alabel);
            fprintf('%s\n', tag);
            fprintf('  A =\n'); disp(A);
            % try
                GenericElliptic_QPDE(fEll, A, N, dim, uTrueEll);
                results(test_id).status = 'PASS';
            % catch ME
            %     results(test_id).status = 'FAIL';
            %     results(test_id).error  = ME.message;
            %     fprintf('  !! ERROR: %s\n', ME.message);
            % end
            results(test_id).tag    = tag;
            results(test_id).solver = 'Elliptic';
            results(test_id).dim    = dim;
            results(test_id).N      = N;
            results(test_id).Alabel = Alabel;
            fprintf('  Status: %s\n\n', results(test_id).status);

        end % iA
    end % iN
end % iDim

%% =========================================================
%  SUMMARY TABLE
% ==========================================================
fprintf('======================================================\n');
fprintf('  SUMMARY\n');
fprintf('======================================================\n');
fprintf('%-5s | %-10s | %-5s | %-6s | %-14s | %s\n', ...
        'Test', 'Solver', 'dim', 'N', 'A', 'Status');
fprintf('%s\n', repmat('-', 1, 62));

n_pass = 0; n_fail = 0;
for k = 1:numel(results)
    pass = strcmp(results(k).status, 'PASS');
    if pass; n_pass = n_pass + 1; else; n_fail = n_fail + 1; end
    fprintf('%-5d | %-10s | %-5d | %-6d | %-14s | %s\n', ...
            k, results(k).solver, results(k).dim, results(k).N, ...
            results(k).Alabel, results(k).status);
    if ~pass
        fprintf('Error: %s\n', results(k).error);
    end
end

fprintf('%s\n', repmat('-', 1, 62));
fprintf('TOTAL: %d/%d passed  |  %d failed\n', n_pass, n_pass+n_fail, n_fail);
fprintf('======================================================\n');