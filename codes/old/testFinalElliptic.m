N=16;
d=2;
A=eye(d);
dt=1e-3;
steps=100;
if d==3
f = @(x,y,z) cos(2*pi*x) .* sin(-4*pi*y).*cos(2*pi*z);

else
f = @(x,y) cos(2*pi*x) .* sin(-4*pi*y);
end

% GenericElliptic_QPDE(f, A, N, d)
GenericDiffusion_QPDE(f, A, N, d,dt,steps)