function f = fftfreq(N, d)
%   f = fftfreq(N, d) returns frequency bins for FFT
%
%   N : number of points
%   d : sample spacing

  if mod(N,2) == 0
    f = [0:(N/2-1), -N/2:-1] / (N*d);
  else
    f = [0:((N-1)/2), -((N-1)/2):-1] / (N*d);
  end
end
