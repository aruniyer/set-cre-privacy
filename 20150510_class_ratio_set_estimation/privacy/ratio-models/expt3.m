function expt3(dataset, train_theta, test_theta, outfile)
fid = fopen(outfile, 'a+');
delta = 0.01;
epsilons = [0.01, 0.1, 0.2];
for i = 1:length(epsilons)
    epsilon = epsilons(i);
    for seed = 1:10
        [psvm_props_lap, psvm_props_gauss, mmd_props_lap, mmd_props_gauss, psvm_acc_lap, psvm_acc_gauss] = run_psvm_mmd_noisy(dataset, seed, train_theta, test_theta, epsilon, delta);
        fprintf(fid,'%s\t%s\t%d\t%s\t%.2f\t%.2f\t%.2f\t%.2f\t%0.2f\t%.2f\t%.2f\n', dataset, 'pSVM', seed, 'laplace', epsilon, delta, train_theta, 1-train_theta, test_theta, abs(psvm_props_lap(1) - test_theta), psvm_acc_lap);
        fprintf(fid,'%s\t%s\t%d\t%s\t%.2f\t%.2f\t%.2f\t%.2f\t%0.2f\t%.2f\t%.2f\n', dataset, 'MMD', seed, 'laplace', epsilon, delta, train_theta, 1-train_theta, test_theta, abs(mmd_props_lap(1) - test_theta), 0);
        fprintf(fid,'%s\t%s\t%d\t%s\t%.2f\t%.2f\t%.2f\t%.2f\t%0.2f\t%.2f\t%.2f\n', dataset, 'pSVM', seed, 'gaussian', epsilon, delta, train_theta, 1-train_theta, test_theta, abs(psvm_props_gauss(1) - test_theta), psvm_acc_gauss);
        fprintf(fid,'%s\t%s\t%d\t%s\t%.2f\t%.2f\t%.2f\t%.2f\t%0.2f\t%.2f\t%.2f\n', dataset, 'MMD', seed, 'gaussian', epsilon, delta, train_theta, 1-train_theta, test_theta, abs(mmd_props_gauss(1) - test_theta), 0);
    end
end