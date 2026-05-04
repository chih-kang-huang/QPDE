%% Generalized Spectral Method Setup (d-dimensions)
d = 3;          % Dimension
n = 4;          % Bits per dimension (N = 2^n)
N = 2^n;
L_dom = 1;      % Domain length
dx = L_dom / N;

% 1. Create 1D spectral components
% Assuming spectral_eigenvalues returns the 1D derivative Fourier multipliers
ev_1D = spectral_eigenvalues(N); 
D = diag(ev_1D);
I = eye(N);

% 2. Build d-dimensional Elliptic operator in spectral space
% We represent the d-dimensional operator as a sum of Kronecker products
A = eye(d); % Coefficient matrix
Elliptic_spec_mat = zeros(N^d, N^d);

for i = 1:d
    for j = 1:d
        if A(i,j) == 0, continue; end
        
        % Build the kron product for (d/dxi * d/dxj)
        term = 1;
        for k = 1:d
            if k == i && k == j
                term = kron(term, D^2);
            elseif k == i || k == j
                term = kron(term, D);
            else
                term = kron(term, I);
            end
        end
        Elliptic_spec_mat = Elliptic_spec_mat + A(i,j) * term;
    end
end

% 3. Handle the zero-frequency mode (Singularity)
% In spectral space, the (1,1,1...) mode is usually the first element
Elliptic_spec_mat(1,1) = 1; 
invElliptic = diag(1 ./ diag(Elliptic_spec_mat));
DiagEncoding = MakeUnitary(invElliptic);

% 4. Quantum Circuit Assembly
% total qubits = d * n (for data) + 1 (ancilla for Unitary encoding)
num_qubits = d * n; 
FG = GroupFourier(d, n); % Your d-dimensional Fourier Group
GF = FG.ctranspose();

totalCircuit = qclab.QCircuit(num_qubits + 1);
totalCircuit.push_back(GF);
% Apply the diagonal scaling in the Fourier domain
totalCircuit.push_back(qclab.qgates.MatrixGate(0:num_qubits, DiagEncoding, "Diagonal"));
totalCircuit.push_back(FG);

%% 5. Generalized RHS (f)
% Create d-dimensional grid
grid_args = cell(1, d);
x_1d = (0:N-1) * dx;
[grid_args{:}] = ndgrid(x_1d); 

% Example: f = prod(sin(2*pi*x_i))
f_val = 1;
for i = 1:d
    f_val = f_val .* sin(2*pi*grid_args{i});
end
f_flatten = f_val(:);

% 6. Apply Operator
totalMat = totalCircuit.matrix;
% Extract the top-left block (accounting for the ancilla)
res = totalMat(1:N^d, 1:N^d) * f_flatten;

% Reshape result back to d-dimensional tensor
reshaped_res = reshape(res, repmat(N, 1, d));