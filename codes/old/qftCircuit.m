

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
