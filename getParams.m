function params = getParams(N, d)
    x_lb = 0;
    x_rb = 1;
    L    = x_rb - x_lb;

    params.dx    = L / N;
    params.n     = log2(N);
    params.N_dir = repmat(N, 1, d);

    x_vec = x_lb + (0:N-1) * params.dx;
    ndgrid_inputs = repmat({x_vec}, 1, d);

    params.grids = cell(1, d);
    [params.grids{:}] = ndgrid(ndgrid_inputs{:});
end