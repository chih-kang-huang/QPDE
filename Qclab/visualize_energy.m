function visualize_energy(energy_class, energy_quant, varargin)
% visualize_energy(E_class, E_quant, dt) 
% OR 
% visualize_energy(E_class, E_quant, E_gt, dt)
%
% Plots the log-energy evolution over time.
set(groot, 'defaultAxesFontSize', 18)
    % 1. Determine if Ground Truth was provided based on number of arguments
    if nargin == 4
        energy_gt = varargin{1};
        dt        = varargin{2};
        has_gt    = true;
    else
        dt        = varargin{1};
        has_gt    = false;
    end

    % 2. Setup time vector
    time = dt * (0:length(energy_class)-1);
    
    figure;
    hold on;

    % 3. Plot Quantum Energy
    plot(time, log(energy_quant), 'LineWidth', 1.5, ...
         'DisplayName', '$E(Q)-E_{\infty}$');
    
    % 4. Plot Classical Energy
    plot(time, log(energy_class), '--', 'LineWidth', 1.5, ...
         'DisplayName', '$E(FG)-E_{\infty}$');
         
    % 5. Plot Ground Truth (if provided)
    if has_gt
        plot(time, log(energy_gt), ':', 'Color', [0.5 0.5 0.5], 'LineWidth', 2, ...
             'DisplayName', '$E(GT)-E_{\infty}$');
    end
    
    hold off;
    
    % Formatting
    ylabel('log(Energy-E_{\infty})');
    xlabel('Time');
    grid on;
    legend('Interpreter', 'latex', 'Location', 'best');
    title('Energy Evolution (Log Scale)');
    
    % Optional: save the figure
    % saveas(gcf, 'Heat2D_energy.png');
end