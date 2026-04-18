function h = make_f_ell(dim)
    switch dim
        case 2; h = @(x,y)   cos(2*pi*x) .* sin(-4*pi*y);
        case 3; h = @(x,y,z) 5*sin(2*pi*x) .* sin(2*pi*y) .* sin(2*pi*z);
    end
end