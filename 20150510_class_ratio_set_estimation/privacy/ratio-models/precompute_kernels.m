function precompute_kernels(dataset)
data_path = strcat('ratio-models/datasets/', dataset);
fprintf(1, 'Datapath = %s\n', data_path);
[X, Y] = dataloader(strcat(data_path, '/full.ssv'));
fprintf(1, 'data loaded.\n');
K = pdist2(X, X);
fprintf(1, 'distances calculated.\n');
K1 = exp(-1*K);
save(strcat(data_path, '/kernel_gamma_1.mat'), 'K1', '-v7.3');
fprintf(1, 'kernel1 saved.\n');
K1 = exp(-0.1*K);
save(strcat(data_path, '/kernel_gamma_0p1.mat'), 'K1', '-v7.3');
fprintf(1, 'kernel2 saved.\n');
K1 = exp(-0.01*K);
save(strcat(data_path, '/kernel_gamma_0p01.mat'), 'K1', '-v7.3');
fprintf(1, 'kernel3 saved.\n');