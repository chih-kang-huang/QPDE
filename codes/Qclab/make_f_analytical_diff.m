function h = make_f_analytical_diff(dim)
    switch dim
        case 2; h = @(x,y)   cos(2*pi*x) .* sin(-4*pi*y);
        case 3; h = @(x,y,z) cos(2*pi*x) .* sin(-4*pi*y) .* cos(2*pi*z);
    end
end