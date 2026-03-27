function reportMetrics(u_class, u_quant, E_class, E_quant, hasGT, gt, E_gt)
    % Flatten spatial arrays to compute the L2 norm over the entire grid
    u_class_flat = u_class(:);
    u_quant_flat = u_quant(:);
    
    fprintf('\n================================================\n');
    fprintf('               SIMULATION METRICS               \n');
    fprintf('================================================\n');
    
    % Check hasGT and ensure gt is actually provided and not empty
    if hasGT && ~isempty(gt)
        gt_flat = gt(:);
        rel_err_classical = norm(u_class_flat - gt_flat) / norm(gt_flat);
        rel_err_quantum   = norm(u_quant_flat - gt_flat) / norm(gt_flat);
        
        fprintf('--- Relative Errors (vs Ground Truth) ---\n');
        fprintf('Classical to GT :  %.4e\n', rel_err_classical);
        fprintf('Quantum to GT   :  %.4e\n', rel_err_quantum);
    else
        % If no ground truth exists, compare Quantum against Classical
        rel_err_quantum = norm(u_quant_flat - u_class_flat) / norm(u_class_flat);
        
        fprintf('--- Relative Error (vs Classical Solution) ---\n');
        fprintf('Quantum to Class:  %.4e\n', rel_err_quantum);
    end
    
    fprintf('\n--- Energy History Summary ---\n');
    fprintf('Classical Energy:  Initial = %.4e  |  Final = %.4e\n', E_class(1), E_class(end));
    fprintf('Quantum Energy  :  Initial = %.4e  |  Final = %.4e\n', E_quant(1), E_quant(end));
    
    % Print Ground Truth Energy if it was provided
    if hasGT && nargin >= 7 && ~isempty(E_gt)
        fprintf('Ground Truth En.:  Initial = %.4e  |  Final = %.4e\n', E_gt(1), E_gt(end));
    end
    
    fprintf('================================================\n\n');
end