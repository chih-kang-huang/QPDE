function reportMetrics(u_class, u_quant, E_class, E_quant, hasGT, gt)
    u_class_flat = u_class(:);
    u_quant_flat = u_quant(:);
    
    if hasGT
        gt_flat = gt(:);
        rel_err_classical = norm(u_class_flat - gt_flat) / norm(gt_flat);
        rel_err_quantum   = norm(u_quant_flat - gt_flat) / norm(gt_flat);
        fprintf('\n--- Relative Errors (vs Ground Truth) ---\n');
        fprintf('Classical Relative Error: %.4e\n', rel_err_classical);
        fprintf('Quantum   Relative Error: %.4e\n', rel_err_quantum);
    else
        rel_err_quantum = norm(u_quant_flat - u_class_flat) / norm(u_class_flat);
        fprintf('\n--- Relative Error (vs Classical Solution) ---\n');
        fprintf('Quantum Relative Error: %.4e\n', rel_err_quantum);
    end
    
    fprintf('\n--- Energy History Summary ---\n');
    fprintf('Classical: initial = %.4e,  final = %.4e\n', E_class(1), E_class(end));
    fprintf('Quantum:   initial = %.4e,  final = %.4e\n', E_quant(1), E_quant(end));
end