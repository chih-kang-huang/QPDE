function plotComparison3D(gt, u_cl, u_qu)
    nx=size(u_cl,1); ny=size(u_cl,2); nz=size(u_cl,3);
    figure('Color','w','Position',[100,100,1400,800]);

    subplot(2,3,1); sliceVol(gt,        nx,ny,nz); title('Ground Truth');           xlabel('x'); ylabel('y'); zlabel('z');
    subplot(2,3,2); sliceVol(u_cl,      nx,ny,nz); title('Classical Solution');     xlabel('x'); ylabel('y'); zlabel('z');
    subplot(2,3,3); sliceVol(abs(gt-u_cl),nx,ny,nz); title('Classical Absolute Error'); xlabel('x'); ylabel('y'); zlabel('z');
    subplot(2,3,5); sliceVol(u_qu,      nx,ny,nz); title('Quantum Solution');       xlabel('x'); ylabel('y'); zlabel('z');
    subplot(2,3,6); sliceVol(abs(gt-u_qu),nx,ny,nz); title('Quantum Absolute Error');   xlabel('x'); ylabel('y'); zlabel('z');

    sgtitle('3D Comparison: Ground Truth, Classical, and Quantum Solutions');
end
