function visualize_simulation_results(u_generic, u_quantum, d, N, ground_truth)
    flag = (nargin >= 5);
    fs   = 14;

    %% ── Output folder ────────────────────────────────────────────────────────
    out_dir = 'matalbfig';
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    persistent fig_counter;
    if isempty(fig_counter)
        fig_counter = 1;
    end

    dim_tag = sprintf('d%d', d);   % 'd2' or 'd3'

    %% ── 3D ──────────────────────────────────────────────────────────────────
    if d == 3
        u_class = reshape(real(u_generic), N, N, N);
        u_quant = reshape(real(u_quantum), N, N, N);
        nx = N; ny = N; nz = N;

        if flag
            hFig = figure('Units', 'normalized', 'Position', [0.1 0.1 0.6 0.6]);
            t = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
            gt3 = reshape(real(ground_truth), N, N, N);

            % Row 1 : Ground Truth | Classical | Quantum
            ax11 = plot3d_panel(1, gt3,     'Ground Truth', nx, ny, nz, fs);
            ax12 = plot3d_panel(2, u_class, 'Classical',    nx, ny, nz, fs);
            ax13 = plot3d_panel(3, u_quant, 'Quantum',      nx, ny, nz, fs);

            clim1 = clim_of(gt3, u_class, u_quant);
            set([ax11 ax12 ax13], 'CLim', clim1);
            format_colorbar(colorbar(ax13), fs);

            % Row 2 : (blank) | Classical Error | Quantum Error
            err_c = abs(gt3 - u_class);
            ax22  = plot3d_panel(5, err_c, 'Classical Error', nx, ny, nz, fs);
            max_err_c = max(err_c(:));
            if max_err_c == 0; max_err_c = 1; end
            set(ax22, 'CLim', [0, max_err_c]);
            format_colorbar(colorbar(ax22), fs);

            err_q = abs(gt3 - u_quant);
            ax23  = plot3d_panel(6, err_q, 'Quantum Error', nx, ny, nz, fs);
            max_err_q = max(err_q(:));
            if max_err_q == 0; max_err_q = 1; end
            set(ax23, 'CLim', [0, max_err_q]);
            format_colorbar(colorbar(ax23), fs);

        else
            hFig = figure('Units', 'normalized', 'Position', [0.1 0.25 0.6 0.3]);
            t = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

            ax1 = plot3d_panel(1, u_class,                'Classical',      nx, ny, nz, fs);
            ax2 = plot3d_panel(2, u_quant,                'Quantum',        nx, ny, nz, fs);
            ax3 = plot3d_panel(3, abs(u_class - u_quant), 'Absolute Error', nx, ny, nz, fs);

            clim1 = clim_of(u_class, u_quant);
            set([ax1 ax2], 'CLim', clim1);
            format_colorbar(colorbar(ax2), fs);
            format_colorbar(colorbar(ax3), fs);
        end


        save_fig(hFig, out_dir, fig_counter, dim_tag);
        fig_counter = fig_counter + 1;
        return
    end

    %% ── 2D ──────────────────────────────────────────────────────────────────
    u_quantum = reshape(real(u_quantum), N, N);

    if flag
        %% ── 6-panel layout ──────────────────────────────────────────────────
        hFig = figure('Units', 'normalized', 'Position', [0.1 0.1 0.6 0.6]); 
        t = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
        
        % Row 1 : Ground Truth | Classical | Quantum
        ax11 = nexttile(1); img2d(ax11, real(ground_truth), 'Ground Truth', fs);
        ax12 = nexttile(2); img2d(ax12, real(u_generic),    'Classical',    fs);
        ax13 = nexttile(3); img2d(ax13, u_quantum,          'Quantum',      fs);
        
        clim1 = clim_of(real(ground_truth), real(u_generic), u_quantum);
        set([ax11 ax12 ax13], 'CLim', clim1);
        format_colorbar(colorbar(ax13), fs); 

        % Row 2 : (blank) | Classical Error | Quantum Error
        ax22 = nexttile(5); 
        err_c = abs(real(ground_truth) - real(u_generic));
        img2d(ax22, err_c, 'Classical Error', fs);
        
        max_err_c = max(err_c(:));
        if max_err_c == 0
            max_err_c = 1;
        end
        set(ax22, 'CLim', [0, max_err_c]);
        format_colorbar(colorbar(ax22), fs); 
        
        ax23 = nexttile(6);
        err_q = abs(real(ground_truth) - u_quantum);
        img2d(ax23, err_q, 'Quantum Error',   fs);
        
        max_err_q = max(err_q(:));
        if max_err_q == 0
            max_err_q = 1;
        end
        set(ax23, 'CLim', [0, max_err_q]);
        format_colorbar(colorbar(ax23), fs);

    else
        %% ── 1-row layout : Classical | Quantum [cb1] | Error [cb2] ──────────
        hFig = figure('Units', 'normalized', 'Position', [0.1 0.25 0.6 0.3]);
        t = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
        
        ax1 = nexttile(1); img2d(ax1, real(u_generic), 'Classical',     fs);
        ax2 = nexttile(2); img2d(ax2, u_quantum,       'Quantum',       fs);
        
        clim1 = clim_of(real(u_generic), u_quantum);
        set([ax1 ax2], 'CLim', clim1);
        format_colorbar(colorbar(ax2), fs); 
        
        ax3 = nexttile(3);
        err = abs(real(u_generic) - u_quantum);
        img2d(ax3, err, 'Absolute Error', fs);
        format_colorbar(colorbar(ax3), fs); 
    end


    save_fig(hFig, out_dir, fig_counter, dim_tag);
    fig_counter = fig_counter + 1;
end

%% ── Save helper ──────────────────────────────────────────────────────────────
function save_fig(hFig, out_dir, counter, dim_tag)
    fname = sprintf('fig%03d_%s.fig', counter, dim_tag);
    savefig(hFig, fullfile(out_dir, fname));
end

%% ── Local helpers ────────────────────────────────────────────────────────────
function img2d(ax, data, ttl, fs)
    imagesc(ax, data); 
    title(ax, ttl, 'FontSize', fs);
    set(ax, 'YDir', 'normal', 'FontSize', fs, ...
        'XTickLabel', [], 'YTickLabel', [], ...
        'DataAspectRatio', [1 1 1]);   
end

function cl = clim_of(varargin)
    vals = cellfun(@(x) x(:), varargin, 'UniformOutput', false);
    vals = vertcat(vals{:});
    cl   = [min(vals), max(vals)];
end

% Returns the axis handle; colorbar management is left to the caller.
function ax = plot3d_panel(tile_idx, data, ttl, nx, ny, nz, fs)
    ax = nexttile(tile_idx);
     slice(ax, data, 1:nx, [], []);hold(ax, 'on')
    slice(ax, data, [], 1:ny, [])
     slice(ax, data, [], [], 1:nz);
     
    shading(ax, 'interp'); axis(ax, 'equal'); axis(ax, 'tight');
    title(ax, ttl, 'FontSize', fs);
    view(ax, 3);
    set(ax, 'FontSize', fs, 'XTickLabel', [], 'YTickLabel', [], 'ZTickLabel', []);
    hold(ax, 'off')
end

function format_colorbar(cb, fs)
    cb.FontSize = fs;
    
    ticks = linspace(cb.Limits(1), cb.Limits(2), 3);
    
    max_val = max(abs(cb.Limits));
    tol = 1e-18 * max_val;
    if max_val == 0
        tol = 1e-18;
    end
    ticks(abs(ticks) < tol) = 0;
    cb.Ticks = ticks;
    
    if max_val > 0
        exponent = floor(log10(max_val));
    else
        exponent = 0;
    end
    multiplier = 10^exponent;
    
    labels = cell(1, 3);
    for i = 1:3
        if ticks(i) == 0
            labels{i} = '0';
        else
            labels{i} = sprintf('%.0f', ticks(i) / multiplier); 
        end
    end
    cb.TickLabels = labels;
    
    if exponent ~= 0
        cb.Title.String = sprintf('\\times 10^{%d}', exponent);
        cb.Title.Interpreter = 'tex';
        cb.Title.FontSize = fs;
    else
        cb.Title.String = '';
    end
end