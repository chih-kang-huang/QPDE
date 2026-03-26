function totalMat = QPDE_Generator_Diffusion(A, n, dt)
% Generates the quantum diffusion operator.
% Inputs:
%   A  : Diffusion coefficient matrix (d x d)
%   n  : Number of qubits per dimension (N = 2^n)
%   dt : Time step size
addpath("/home/g.antonioli/qclab")
d = size(A, 1);
N = 2^n;

% --- 1. Build Spectral Operator ---
N_vecs = repmat(N, 1, d); % Grid size for each dimension
dx     = 1.0 / N;         % Assuming domain length L = 1.0


OP_vals = buildDiffusionDenom(A, N_vecs, dx, d);

fprintf("OPVALS(1,1):\n")
disp(OP_vals(1,1));

Inv_OP_vals = 1 ./ (1 - dt * OP_vals(:));

alpha = max(abs(Inv_OP_vals));
Normalized_Vals = Inv_OP_vals / alpha;


DiagMat = sparse(1:N^d, 1:N^d, Normalized_Vals, N^d, N^d);


DiagEncoding = MakeUnitary(DiagMat);


FG = GroupFourier(d, n);
GF = FG.ctranspose();

totalCircuit = qclab.QCircuit(d*n + 1);
totalCircuit.push_back(GF);
totalCircuit.push_back(qclab.qgates.MatrixGate(0:d*n, DiagEncoding, "Diagonal"));
totalCircuit.push_back(FG);

M = totalCircuit.matrix;
totalMat = real(M(1:2^(d*n), 1:2^(d*n))) * alpha;

end