
%%
% Define normalized constants
c1 = 1/sqrt(8);
c2 = 1/2;

% Define GFT matrix FG
FG = [
    c1,  c1,  c1,  c1,  c1,  c1,  c1,  c1;
    c1,  c1,  c1,  c1, -c1, -c1, -c1, -c1;
    c1, -c1,  c1, -c1,  c1, -c1,  c1, -c1;
    c1, -c1,  c1, -c1, -c1,  c1, -c1,  c1;
    c2, -c2*1i, -c2,  c2*1i, 0, 0, 0, 0;
    0, 0, 0, 0, c2, c2*1i, -c2, -c2*1i;
    0, 0, 0, 0, c2, -c2*1i, -c2, c2*1i;
    c2, c2*1i, -c2, -c2*1i, 0, 0, 0, 0
];

% Inverse GFT
GF = FG';

% Define signals
f = [1; 0; 0; 0; 0; -1i; 1i; 1];
m = f;

% Transform to spectral domain
m_h = FG * m;
f_h = FG * f;

% Theoretical convolution in spectral domain
prod_m_h_f_h = [
    m_h(1) * f_h(1);
    m_h(2) * f_h(2);
    m_h(3) * f_h(3);
    m_h(4) * f_h(4);
    reshape(reshape(m_h(5:8), 2, 2) * reshape(f_h(5:8), 2, 2), [], 1)
];

theoretical_res = GF * prod_m_h_f_h;

% Direct sum functions
function result = direct_sum(A, B)
    [m, n] = size(A);
    [p, q] = size(B);
    result = zeros(m + p, n + q);
    result(1:m, 1:n) = A;
    result(m+1:end, n+1:end) = B;
end

function result = direct_sum_all(varargin)
    result = zeros(0, 0);
    for k = 1:length(varargin)
        result = direct_sum(result, varargin{k});
    end
end

% Spectral filter
spectral_filter = direct_sum_all( ...
    m_h(1) * eye(1), ...
    m_h(2) * eye(1), ...
    m_h(3) * eye(1), ...
    m_h(4) * eye(1), ...
    kron(eye(2), reshape(m_h(5:8), 2, 2)) ...
);

% Apply filter and inverse transform
res = GF * spectral_filter * f_h;

% Compare with theoretical result
error_norm = norm(res - theoretical_res);
disp(['Norm of the difference: ', num2str(error_norm)]);


%%
% Define normalized constants
c1 = 1/sqrt(8);
c2 = 1/2;

% Define GFT
FG = [
    c1,  c1,  c1,  c1,  c1,  c1,  c1,  c1;
    c1,  c1,  c1,  c1, -c1, -c1, -c1, -c1;
    c1, -c1,  c1, -c1,  c1, -c1,  c1, -c1;
    c1, -c1,  c1, -c1, -c1,  c1, -c1,  c1;
    c2, -c2*1i, -c2,  c2*1i, 0, 0, 0, 0;
    0, 0, 0, 0, c2, c2*1i, -c2, -c2*1i;
    0, 0, 0, 0, c2, -c2*1i, -c2, c2*1i;
    c2, c2*1i, -c2, -c2*1i, 0, 0, 0, 0
];

GF=FG';

QGT0=qclab.QCircuit(3);
QGT0.push_back(qclab.qgates.MatrixGate([0,1,2],FG));
QGT1=qclab.QCircuit(3,3);
QGT1.push_back(qclab.qgates.MatrixGate([0,1,2],FG));

QGT=qclab.QCircuit(3,1);
QGT.push_back(QGT0);
%QGT.push_back(QGT1);



QGT.draw()
test=QGT.matrix;
norm(FG-test)
invQGT=QGT.ctranspose();
test1=invQGT.matrix();
norm(GF-test1)

DM=diag(m)
alpha=norm(DM);
DAIG=DM/alpha;
unitary=MakeUnitary(DAIG);

BlockEncoding=qclab.QCircuit(log2(size(unitary,1)))
BlockEncoding.push_back(qclab.qgates.MatrixGate(0:log2(size(unitary,1))-1,unitary))
BlockEncoding.draw()

totalCirc=qclab.QCircuit(log2(size(unitary,1)))
totalCirc.push_back(invQGT)
totalCirc.push_back(BlockEncoding)
totalCirc.push_back(QGT)
totalCirc.draw()
tmp=totalCirc.matrix;
A=tmp(1:8,1:8);
myexpected=FG*DM*GF;
norm(myexpected-A*alpha)