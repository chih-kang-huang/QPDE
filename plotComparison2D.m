function plotComparison2D(gt, u_cl, u_qu)
    subplot(2,3,1); imagesc(gt');          axis square; colorbar; title('Ground Truth');        xlabel('x'); ylabel('y'); set(gca,'YDir','normal');
    subplot(2,3,2); imagesc(real(u_cl)');  axis square; colorbar; title('Classical Solution');  xlabel('x'); ylabel('y'); set(gca,'YDir','normal');
    subplot(2,3,3); imagesc(abs(gt-u_cl)');axis square; colorbar; title('Classical Error');     xlabel('x'); ylabel('y'); set(gca,'YDir','normal');
    subplot(2,3,5); imagesc(u_qu');        axis square; colorbar; title('Quantum Solution');    xlabel('x'); ylabel('y'); set(gca,'YDir','normal');
    subplot(2,3,6); imagesc(abs(gt-u_qu)');axis square; colorbar; title('Quantum Error');       xlabel('x'); ylabel('y'); set(gca,'YDir','normal');
end