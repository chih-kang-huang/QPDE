function sliceVol(V, nx, ny, nz)
    % Shared 3-axis slice rendering used by all 3D plots
    slice(V, [], [], 1:nz); hold on;
    slice(V, 1:nx, [], []);
    slice(V, [], 1:ny, []);
    shading interp; axis equal tight; colorbar; view(3); hold off;
end