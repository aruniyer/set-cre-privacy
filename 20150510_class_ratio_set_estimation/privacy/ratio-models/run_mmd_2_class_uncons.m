function run_mmd_2_class_uncons(dataset, outfile)

% prefix directory
data_path = strcat('./datasets/', dataset);

% create output file
outfile_path = strcat(data_path, '/', outfile);
fid = fopen(outfile_path, 'w+');

% load dataset
[X, Y] = dataloader(strcat(data_path, '/full.ssv'));

% load pre-computed kernels
Ks = cell(3, 1);
K = load(strcat(data_path, '/kernel_gamma_1.mat'));
Ks{1} = K.K1;
K = load(strcat(data_path, '/kernel_gamma_0p1.mat'));
Ks{2} = K.K1;
K = load(strcat(data_path, '/kernel_gamma_0p01.mat'));
Ks{3} = K.K1;

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

% for every seed, every proportion run mmd
for seed = 1:10
    for test_theta = 0.1:0.1:0.9
        test_set_proportion = [test_theta, 1-test_theta];
        split = get_splits(Y, seed, training_set_sizes,...
            training_set_proportions, testing_set_size,...
            test_set_proportion);
        
        kernel = Ks{mmd_parameters(seed, 1)};
        mmd_props = MMD_uncons(kernel, split, []);
        train_theta = 0.1;
        fprintf(fid,'%s\t%s\t%d\t%.2f\t%.2f\t%0.2f\t%.2f\t%.2f\n', dataset, 'MMD_uncons', seed, train_theta, 1-train_theta, test_theta, abs(mmd_props(1) - test_theta), 0);
    end
end
fclose(fid);