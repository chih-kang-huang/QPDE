function totalMat = QPDE_Generator_helmoltz(k,d, n)
% Builds the quantum circuit matrix for the elliptic PDE solver.
%
%   Inputs:
%     A - d x d diffusion coefficient matrix
%     n - number of qubits per dimension (grid size N = 2^n)
%
%   Output:
%     totalMat - 2^(d*n) x 2^(d*n) unitary operator matrix
% addpath("/home/g.antonioli/qclab")
N = 2^n;


N_vecs = repmat(N, 1, d);
dx     = 1.0 / N;         


denom = buildHelmoltzDenom(k, N_vecs, dx, d);

invOP = diag(1 ./ denom(:));
alpha=norm(invOP,'inf')
invOP=invOP/alpha;
DiagEncoding = MakeUnitary(invOP);

FG = GroupFourier(d, n);
GF = FG.ctranspose();

totalCircuit = qclab.QCircuit(d*n + 1);
totalCircuit.push_back(GF);
totalCircuit.push_back(qclab.qgates.MatrixGate(0:d*n, DiagEncoding, "Diagonal"));
totalCircuit.push_back(FG);

M        = totalCircuit.matrix;
totalMat = M(1:2^(d*n), 1:2^(d*n))*alpha;

end