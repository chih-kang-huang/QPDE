function k = spectral_eigenvalues(N, L)
%   Eigenvalues of d/dx with periodic BCs: 2*pi*i*freq
%
%   Inputs:
%     N - number of grid points
%     L - domain length (default: 1.0)
%
%   Output:
%     k - row vector of Fourier eigenvalues (2π i k / L)

    if nargin < 2
        L = 1.0;
    end

    freq = fftfreq(N, L/N);
    k    = 2i * pi * freq;
end

function f = fftfreq(N, d)
%   Standard FFT frequency vector.
%
%   Inputs:
%     N - number of points
%     d - sample spacing
%
%   Output:
%     f - frequency vector of length N

    if mod(N, 2) == 0
        f = [0:(N/2-1), -N/2:-1] / (N * d);
    else
        f = [0:((N-1)/2), -((N-1)/2):-1] / (N * d);
    end
end