function saveResults_Diffusion(u_generic, u_quantum, E_class, E_quant, d, flagUtrue, ground_truth, E_gt)
    % Handle missing arguments safely
    if nargin < 7
        ground_truth = [];
    end
    if nargin < 8
        E_gt = [];
    end
    
    % Setup directory and filename
    timestamp  = char(datetime('now', 'Format', 'yyyyMMdd_HHmmSS'));
    folderName = sprintf('Results/Results_d%d_%s', d, timestamp);
    if ~exist(folderName, 'dir')
        mkdir(folderName);
    end
    
    h5file = fullfile(folderName, 'simulation_data.h5');
    u_q_real = double(real(u_quantum));
    
    % Save Solutions
    h5create(h5file, '/u_generic', size(u_generic));
    h5write( h5file, '/u_generic', double(real(u_generic)));
    
    h5create(h5file, '/u_quantum', size(u_q_real));
    h5write( h5file, '/u_quantum', u_q_real);
    
    % Save Energy Histories
    h5create(h5file, '/E_class_history', size(E_class));
    h5write( h5file, '/E_class_history', double(E_class));
    
    h5create(h5file, '/E_quant_history', size(E_quant));
    h5write( h5file, '/E_quant_history', double(E_quant));
    
    % Save Ground Truth Data (if applicable)
    if flagUtrue && ~isempty(ground_truth) && ~isempty(E_gt)
        % Save Spatial Ground Truth
        h5create(h5file, '/u_true', size(ground_truth));
        h5write( h5file, '/u_true', double(ground_truth));
        
        % Save Energy Ground Truth
        h5create(h5file, '/E_gt_history', size(E_gt));
        h5write( h5file, '/E_gt_history', double(E_gt));
        
        fprintf('Saved u_generic, u_quantum, energy histories, u_true, and E_gt_history to %s\n', folderName);
    else
        fprintf('Saved u_generic, u_quantum, and energy histories to %s\n', folderName);
    end
end