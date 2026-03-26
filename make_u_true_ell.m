function h = make_u_true_ell(dim)
    switch dim
        case 2; h = @(x,y)  -cos(2*pi*x) .* sin(-4*pi*y) / (20*pi^2);
        case 3; h = @(x,y,z)-cos(2*pi*x) .*sin(-4*pi*y).*cos(pi*z) / (24*pi^2)
            %u_true(x,y,z)=-\cos(2\pi x) \sin(-4\pi y) \cos (2\pi z)/(24\pi^2)
    end
end