function [u_final, history] = QuantumDiffusionSolver(A, N, dt, steps, u_init, f_source)
%QUANTUMDIFFUSIONSOLVER Solves the diffusion equation using a Quantum-mimetic spectral method.
%   [u_final, history] = QuantumDiffusionSolver(A, N, dt, steps, u_init, f_source)
%
%   Inputs:
%       A        : Diffusion tensor (d x d matrix, e.g., 2x2 for 2D)
%       N        : Grid size per dimension (must be power of 2, e.g., 32)
%       dt       : Time step size
%       steps    : Number of time steps to evolve
%       u_init   : Initial condition (N x N matrix)
%       f_source : Source term field (N x N matrix)
%
%   Outputs:
%       u_final  : The solution field at the final time step
%       history  : Structure containing energy history and time points

    %% 1. Parameters & Setup
    dim = size(A, 1);       % Dimensionality (e.g., 2)
    n_qubits = log2(N);     % Qubits per dimension
    
    if mod(n_qubits, 1) ~= 0
        error('Grid size N must be a power of 2.');
    end
    
    % Wave Vectors Construction
    % Corresponds to Python: k = fftfreq(N) * 2pi * 1j
    k_ind = [0:N/2-1, -N/2:-1]';
    k_base = 2i * pi * k_ind;       
    
    % Generate grids for wave vectors (K_grids{1}=kx, K_grids{2}=ky)
    [K_grids{1:dim}] = ndgrid(k_base);

    %% 2. Operator Construction
    % Construct Elliptic Operator in Frequency Domain: div(A * grad)
    % This computes Sum( A_ij * k_i * k_j )
    Elliptic_diag = zeros(size(K_grids{1}));
    for i = 1:dim
        for j = 1:dim
            Elliptic_diag = Elliptic_diag + A(i,j) * K_grids{i} .* K_grids{j};
        end
    end
    
    % Create Semi-Implicit Filter: (1 - dt * Operator)^-1
    Denominator = 1 - dt * Elliptic_diag;
    Filter = 1 ./ Denominator;
    
    % Normalization for Quantum Block Encoding
    % We must ensure the operator norm <= 1 for the unitary encoding
    alpha = max(abs(Filter(:)));           
    Normalized_Filter = Filter(:) / alpha;
    
    % Build Quantum Circuit Components
    % Note: In a real QPU run, this would compile to gates. Here we emulate the matrix.
    FG = GroupFourier(dim, n_qubits);       % Forward QFT
    GF = FG.ctranspose();                   % Inverse QFT
    DiagGate = MakeUnitary(diag(Normalized_Filter)); % Block encoding of diagonal
    
    % Construct Full Operator Matrix: M = iQFT * Diag * QFT
    QC = qclab.QCircuit(dim*n_qubits + 1);
    QC.push_back(FG);
    QC.push_back(qclab.qgates.MatrixGate(0:dim*n_qubits, DiagGate, "D"));
    QC.push_back(GF);
    
    FullMat = QC.matrix;
    % Extract the top-left block acting on the computational basis
    Q_Op = FullMat(1:N^dim, 1:N^dim);

    %% 3. Time Evolution
    u_curr = u_init;
    
    % Energy Helper: E = 0.5 * <u, -div(A grad u)>
    % Scale factor N^(2*dim) accounts for FFT/IFFT normalization differences in Energy calc
    calc_energy = @(u_in) 0.5 * real(sum(sum( conj(fftn(u_in)) .* (-Elliptic_diag) .* fftn(u_in) ))) / N^(2*dim);
    
    history.energy = zeros(1, steps + 1);
    history.energy(1) = calc_energy(u_curr);
    history.time = 0:dt:(steps*dt);
    
    % Flatten source term for vector multiplication
    f_vec = f_source(:);
    
    for t = 1:steps
        % Quantum Step Logic: u_{n+1} = Op * (u_n - dt*f) * alpha
        
        % 1. Form input vector (u - dt*f)
        v_vec = u_curr(:) - dt * f_vec;
        
        % 2. Apply Quantum Operator
        u_next_vec = Q_Op * v_vec;
        
        % 3. Rescale by alpha (decoding)
        u_next_vec = u_next_vec * alpha;
        
        % 4. Reshape back to grid
        u_curr = reshape(real(u_next_vec), repmat(N, 1, dim));
        
        % Record Energy
        history.energy(t+1) = calc_energy(u_curr);
    end
    
    u_final = u_curr;
    history.alpha = alpha; % Store alpha for reference
end