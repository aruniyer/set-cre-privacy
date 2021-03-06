function run_mmd_2_class(dataset, outfile)

% prefix directory
data_path = strcat('ratio-models/datasets/', dataset);
fprintf(1, 'Data path = %s\n', data_path);

% create output file
outfile_path = strcat(data_path, '/', outfile);
fid = fopen(outfile_path, 'w+');
fprintf(1, 'Outfile = %s\n', outfile_path);

% load dataset
[X, Y] = dataloader(strcat(data_path, '/full.ssv'));
fprintf(1, 'Data loaded.\n');

% load pre-computed kernels
Ks = cell(3, 1);
K = load(strcat(data_path, '/kernel_gamma_1.mat'));
Ks{1} = K.K1;
K = load(strcat(data_path, '/kernel_gamma_0p1.mat'));
Ks{2} = K.K1;
K = load(strcat(data_path, '/kernel_gamma_0p01.mat'));
Ks{3} = K.K1;
fprintf(1, 'Kernels loaded.\n');

% set train sizes and proportions
training_set_sizes = [...
    600, 600, 600, 600,...
];
training_set_proportions = [...
    0.1, 0.9;...
    0.9, 0.1;...
    0.1, 0.9;...
    0.9, 0.1;...
];
    
% set test set size
testing_set_size = 600;

% load cross-validated parameters
cvparams = load(strcat(data_path, '/mmd_parameters'));
mmd_parameters = cvparams.mmd_parameters;
fprintf(1, 'Params loaded.\n');

% for every seed, every proportion run mmd
for seed = 1:10
    for test_theta = 0.1:0.1:0.9
        fprintf(1, '%d, %.2f\n', seed, test_theta);
        test_set_proportion = [test_theta, 1-test_theta];
        split = get_splits(Y, seed, training_set_sizes,...
            training_set_proportions, testing_set_size,...
            test_set_proportion);
        
        kernel = Ks{3};
        mmd_props = MMD(kernel, split, []);
        train_theta = 0.1;
        fprintf(fid,'%s\t%s\t%d\t%.2f\t%.2f\t%0.2f\t%.2f\t%.2f\n', dataset, 'MMD', seed, train_theta, 1-train_theta, test_theta, abs(mmd_props(1) - test_theta), 0);
    end
end
fclose(fid);