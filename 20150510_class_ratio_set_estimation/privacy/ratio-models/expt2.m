function expt2(dataset, train_theta, test_thetas, outfile)
fid = fopen(outfile, 'a+');
for seed = 1:10
    [ mmd_prop_mat, psvm_prop_mat, psvm_acc_mat ] = run_psvm_mmd(dataset, seed, train_theta, test_thetas);
    for i = 1:length(test_thetas);
        test_theta = test_thetas(i);
        psvm_props = psvm_prop_mat(i, :);
        mmd_props = mmd_prop_mat(i, :);
        psvm_acc = psvm_acc_mat(i);
        
        fprintf(fid,'%s\t%s\t%d\t%.2f\t%.2f\t%0.2f\t%.2f\t%.2f\n', dataset, 'pSVM', seed, train_theta, 1-train_theta, test_theta, abs(psvm_props(1) - test_theta), psvm_acc);
        fprintf(fid,'%s\t%s\t%d\t%.2f\t%.2f\t%0.2f\t%.2f\t%.2f\n', dataset, 'MMD', seed, train_theta, 1-train_theta, test_theta, abs(mmd_props(1) - test_theta), 0);
    end
end