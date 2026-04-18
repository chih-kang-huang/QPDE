function saveResults(u_generic, u_quantum, d, flagUtrue, ground_truth)
    if nargin < 5
        ground_truth = [];
    end

timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss_SSS'));    
folderName = sprintf('Results/Results_d%d_%s', d, timestamp);

    if ~exist(folderName, 'dir')
        mkdir(folderName);
    end

    h5file = fullfile(folderName, 'simulation_data.h5');
    u_q_real = double(real(u_quantum));

    h5create(h5file, '/u_generic', size(u_generic));
    h5write( h5file, '/u_generic', double(real(u_generic)));
    h5create(h5file, '/u_quantum', size(u_q_real));
    h5write( h5file, '/u_quantum', u_q_real);

    if flagUtrue && ~isempty(ground_truth)
        h5create(h5file, '/u_true', size(ground_truth));
        h5write( h5file, '/u_true', double(ground_truth));
        %fprintf('Saved u_generic, u_quantum, and u_true to %s\n', folderName);
    %end
        %fprintf('Saved u_generic and u_quantum to %s\n', folderName);
    end
end