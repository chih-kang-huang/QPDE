function ground_truth = computeDiffusionGroundTruth(rhs, denom_gt)
    rhs_h        = fftn(rhs);
    ground_truth = real(ifftn(rhs_h .* denom_gt));
end