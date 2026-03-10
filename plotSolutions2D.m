function plotSolutions2D(u_cl, u_qu)
    subplot(1,3,1); imagesc(u_cl');             axis square; colorbar; title('Classical Solution'); xlabel('x'); ylabel('y'); set(gca,'YDir','normal');
    subplot(1,3,2); imagesc(u_qu');             axis square; colorbar; title('Quantum Solution');   xlabel('x'); ylabel('y'); set(gca,'YDir','normal');
    subplot(1,3,3); imagesc(abs(u_cl-u_qu)');   axis square; colorbar; title('Absolute Error');     xlabel('x'); ylabel('y'); set(gca,'YDir','normal');
end