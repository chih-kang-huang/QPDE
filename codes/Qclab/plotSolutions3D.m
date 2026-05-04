function plotSolutions3D(u_cl, u_qu)
    nx=size(u_cl,1); ny=size(u_cl,2); nz=size(u_cl,3);

    subplot(1,3,1); sliceVol(u_cl,         nx,ny,nz); title('Classical Solution'); xlabel('x'); ylabel('y'); zlabel('z');
    subplot(1,3,2); sliceVol(u_qu,         nx,ny,nz); title('Quantum Solution');   xlabel('x'); ylabel('y'); zlabel('z');
    subplot(1,3,3); sliceVol(abs(u_cl-u_qu),nx,ny,nz); title('Absolute Error');    xlabel('x'); ylabel('y'); zlabel('z');
end