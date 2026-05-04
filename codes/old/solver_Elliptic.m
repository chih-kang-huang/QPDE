function u = solver_Elliptic(f, xs, ys, A, N_x, N_y, dx)

  % Evaluate RHS on the grid
  f_values = f(xs, ys);

  % Forward FFT
  f_h = fft2(f_values);

  % Spectral wave numbers (periodic domain)
  kx = 2i*pi*fftfreq(N_x, dx);
  ky = 2i*pi*fftfreq(N_y, dx);

  % Avoid division by zero (mean mode)
  kx(1) = 1;
  ky(1) = 1;

  % 2D spectral grid with ij-indexing
  [k_x, k_y] = ndgrid(kx, ky);

  % Elliptic operator in Fourier space
  denom = ...
      A(1,1)*k_x.^2 + ...
      A(2,1)*k_x.*k_y + ...
      A(1,2)*k_y.*k_x + ...
      A(2,2)*k_y.^2;

  % Solve in spectral space
  u_h = f_h ./ denom;

  % Inverse FFT (solution in physical space)
  u = ifft2(u_h, N_x, N_y);
end
