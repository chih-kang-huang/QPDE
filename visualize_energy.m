function visualize_energy(energy_true, energy_num, dt)
% Plots the log-energy evolution over time.

    time = dt * (0:length(energy_true)-1);

    log_energy_true = log(energy_true);
    log_energy_num  = log(energy_num);
    
    figure;
   
    plot(time, log_energy_num, 'LineWidth', 1.5, ...
         'DisplayName', '$Energy_{quantum}$');
    hold on;
    
    plot(time, log_energy_true, '--', 'LineWidth', 1.5, ...
         'DisplayName', '$Energy_{classical}$');
         
    hold off;
    
    ylabel('log(Energy)');

    xlabel('Time');
    
    legend('Interpreter', 'latex', 'Location', 'best');
    
    % plt.savefig("Heat2D_energy.png", bbox_inches="tight")
    % saveas(gcf, 'Heat2D_energy.png');
end