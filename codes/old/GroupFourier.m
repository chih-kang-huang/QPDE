function totalCircuit = GroupFourier(d,n)
% Total circuit: d registers of size n + 1 ancilla
totalCircuit = qclab.QCircuit(d*n + 1);

% Apply a QFT to each n-qubit register
for k = 1:d
  % Starting qubit index for the k-th register (1-based)
  offset = (k-1)*n + 1;

  % QFT circuit acting on qubits [offset, ..., offset+n-1]
  qft_k = qclab.QCircuit(n, offset);

  % Build the QFT
  qftCircuit(qft_k, n);

  % Append to total circuit
  totalCircuit.push_back(qft_k);
end

% Draw the full circuit
%totalCircuit.draw;

end