warning('off', 'all');

function eigvals = laplacian_eigenvalues(N, L)
    % Eigenvalues of the 1D Laplacian with periodic boundary conditions
    if nargin < 2
        L = 1.0;
    end
    dx = L / N;
    k = (0:N-1)';  % frequency indices
    k(k > N/2) = k(k > N/2) - N;  % shift frequencies for FFT
    k = k * (2 * pi / L);  % scale to physical frequencies
    eigvals = -k.^2;  % Laplacian eigenvalues
end

function Delta_hat = generateD(N)

d=laplacian_eigenvalues(N);

Lambda=diag(d);

Delta_hat = kron(eye(N), Lambda) + kron(Lambda, eye(N));
Delta_hat(1,1) = 1;

Delta_hat=diag(1./diag(Delta_hat));



end


function qftCircuit( circuit, nbQubits )
  H = @qclab.qgates.Hadamard ;
  CP = @qclab.qgates.CPhase ;
  SWAP = @qclab.qgates.SWAP ;
  
  n = double(nbQubits) ;
  % B blocks
  for i = 0 : n - 1
    % Hadamard
    circuit.push_back( H( i ) );
    % diagonal blocks
    for j = 2 : n-i
      control = j + i - 1 ;
      theta = -2*pi/2^j ;
      circuit.push_back( CP( control, i, theta ) ) ;
    end
  end
  
  % swaps
  for i = 0 : floor(n/2) - 1
    circuit.push_back( SWAP( i, n - i - 1 ) );
  end
end



% test for abelian group FT

%DiagGate=qclab.qgates.MatrixGate(tot-(n+1):tot-1,MakeUnitary(Du));


function [final, alpha,mat ] = GenerateConvolution_d2(A)
tic;

[rows,cols]=size(A);
n=log2(rows)/2;
circ=qclab.QCircuit(2*n+1);
fprintf("Generating QFT\n");
circuit = qclab.QCircuit( n,1 ) ;
qftCircuit( circuit ,n);
%circuit.draw()
circuit1 = qclab.QCircuit( n,n+1) ;
qftCircuit( circuit1 ,n);
circ.push_back(circuit);
circ.push_back(circuit1);

fprintf("Normalizing Matrix\n");
alpha=norm(A);
DiagTotal=A/norm(A);
fprintf("Generating Block Encoding of Matrix\n");

tmp=MakeUnitary(DiagTotal);
DiagGate=qclab.qgates.MatrixGate(0:2*n ,tmp);
fprintf("Generating IQFT\n");
inverse_qft=circ.ctranspose();

fprintf("Composing Circuit\n");

final=qclab.QCircuit(2*n+1);
final.push_back(circ);

final.push_back(DiagGate);


final.push_back(inverse_qft);
%final.draw()

fprintf("Extracting Matrix\n");

mat=final.matrix;

FG=kron(dftmtx(2^n),dftmtx(2^n));

GF=kron(conj(dftmtx(2^n))/2^n,conj(dftmtx(2^n))/2^n);


expected=GF*A*FG/norm(A);
%alpha
error=norm(expected-full(mat(1:rows,1:cols)));

fprintf("Error of encoding: %f\n",error);
toc
end




n=4;
N=2^n;
fprintf("Generating Matrix with N: %d\n",N);

DiagTotal=generateD(N);


[final, alpha,mat ] = GenerateConvolution_d2(DiagTotal);


f = @(x,y) cos(2*pi*x)*sin(-2*pi*y);
inputVal=[];
%inputs=[0,1/4, 1/2, 3/4];

 % You can change this to any positive integer
inputs = (0:N-1) / N;

for i=0:N-1
    for j=0:N-1
        inputVal(end+1)=f(inputs(i+1),inputs(j+1));
    end
end
normalized_input=inputVal'/norm(inputVal);


u=@(x,y) -1/(8*pi^2)*cos(2*pi*x)*sin(-2*pi*y);

res=[];
for i=0:N-1
    for j=0:N-1
        res(end+1)=u(inputs(i+1),inputs(j+1));
    end
end



error=norm(res'-full(mat(1:N^2,1:N^2)*norm(DiagTotal)*normalized_input*norm(inputVal)));

fprintf("Error between expected result and actual result: %.15f\n", error);




