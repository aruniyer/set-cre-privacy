function run_psvm_2_class_noisy(dataset, outfile, seeds)

% prefix directory
data_path = strcat('ratio-models/datasets/', dataset);

% create output file
outfile_path = strcat(data_path, '/', outfile);
fid = fopen(outfile_path, 'w+');
fclose(fid);

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
cvparams = load(strcat(data_path, '/psvm_parameters_bak'));
parameters = cvparams.psvm_parameters_bak;

% for every seed, every proportion run mmd
test_thetas = 0.1:0.1:0.9;
parfor (k = 1:length(test_thetas), 8)
    test_theta = test_thetas(k);
    for i = 1:length(seeds)  
        seed = seeds(i);
        test_set_proportion = [test_theta, 1-test_theta];
        split = get_splits(Y, seed, training_set_sizes,...
            training_set_proportions, testing_set_size,...
            test_set_proportion);
        original_copy = split.train_bag_prop;
        
        for j = 1:length(training_set_sizes)
            m = split.train_bag_prop(j) * training_set_sizes(j);
            n = (1 - split.train_bag_prop(j)) * training_set_sizes(j);
            
            % Laplace noise
            noisy_props = laplace_noise(0.05, 0.05, [m; n]);
            split.train_bag_prop(j) = noisy_props(1);            
        end
        kernel = Ks{parameters(seed, 1)};
        mmd_props = pSVM(kernel, split, parameters(2:3));
        fid1 = fopen(outfile_path, 'a+');
        fprintf(fid1,'%s\t%s\t%d\t%s\t%.2f\t%.2f\t%0.2f\t%.2f\t%.2f\n', dataset, 'pSVM', seed, 'laplace', split.train_bag_prop(1), split.train_bag_prop(2), test_theta, abs(mmd_props(1) - test_theta), 0);
        fclose(fid1);
        
        split.train_bag_prop = original_copy;
        for j = 1:length(training_set_sizes)
            m = split.train_bag_prop(j) * training_set_sizes(j);
            n = (1 - split.train_bag_prop(j)) * training_set_sizes(j);
            
            % Gaussian noise
            noisy_props = gaussian_noise(0.05, 0.05, [m; n]);
            split.train_bag_prop(j) = noisy_props(1);            
        end
        kernel = Ks{parameters(seed, 1)};
        mmd_props = pSVM(kernel, split, parameters(2:3));
        fid1 = fopen(outfile_path, 'a+');
        fprintf(fid1,'%s\t%s\t%d\t%s\t%.2f\t%.2f\t%0.2f\t%.2f\t%.2f\n', dataset, 'pSVM', seed, 'gaussian', split.train_bag_prop(1), split.train_bag_prop(2), test_theta, abs(mmd_props(1) - test_theta), 0);
        fclose(fid1);
        
        split.train_bag_prop = original_copy;
        for j = 1:length(training_set_sizes)
            m = split.train_bag_prop(j) * training_set_sizes(j);
            n = (1 - split.train_bag_prop(j)) * training_set_sizes(j);
            
            % Dirichlet noise
            noisy_props = dirichlet_noise(0.05, 0.05, [m; n]);
            split.train_bag_prop(j) = noisy_props(1);            
        end
        kernel = Ks{parameters(seed, 1)};
        mmd_props = pSVM(kernel, split, parameters(2:3));
        fid1 = fopen(outfile_path, 'a+');
        fprintf(fid1,'%s\t%s\t%d\t%s\t%.2f\t%.2f\t%0.2f\t%.2f\t%.2f\n', dataset, 'pSVM', seed, 'dirichlet', split.train_bag_prop(1), split.train_bag_prop(2), test_theta, abs(mmd_props(1) - test_theta), 0);
        fclose(fid1);
    end
end