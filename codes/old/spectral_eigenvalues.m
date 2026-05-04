function k = spectral_eigenvalues(N, L)
%SPECTRAL_EIGENVALUES Eigenvalues of the 1D derivative operator
%   with periodic boundary conditions.
%
%   k = spectral_eigenvalues(N, L)
%
%   Inputs:
%     N - number of grid points
%     L - domain length (default: 1.0)
%
%   Output:
%     k - Fourier eigenvalues (2π i k)

  if nargin < 2
    L = 1.0;
  end

  % FFT frequency vector (MATLAB convention)
  freq = fftfreq(N, L/N);

  % Eigenvalues of the first derivative operator
  k = 2i * pi * freq;
end

function f = fftfreq(N, d)
%FFTFREQ MATLAB equivalent of numpy/jax fftfreq
  if mod(N,2) == 0
    f = [0:(N/2-1), -N/2:-1] / (N*d);
  else
    f = [0:((N-1)/2), -((N-1)/2):-1] / (N*d);
  end
end
