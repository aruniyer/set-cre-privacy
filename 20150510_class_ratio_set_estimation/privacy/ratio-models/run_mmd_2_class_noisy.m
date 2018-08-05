function run_mmd_2_class_noisy(dataset, outfile)

% prefix directory
data_path = strcat('ratio-models/datasets/', dataset);
fprintf(1, 'Datapath = %s\n', data_path);


% create output file
outfile_path = strcat(data_path, '/', outfile);
fid = fopen(outfile_path, 'w+');
fprintf(1, 'Outfile = %s\n', outfile);

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
        original_copy = split.train_bag_prop;
        
        for j = 1:length(training_set_sizes)
            m = split.train_bag_prop(j) * training_set_sizes(j);
            n = (1 - split.train_bag_prop(j)) * training_set_sizes(j);
            
            % Laplace noise
            noisy_props = laplace_noise(0.05, 0.05, [m; n]);
            split.train_bag_prop(j) = noisy_props(1);            
        end
        kernel = Ks{mmd_parameters(seed, 1)};
        mmd_props = MMD_uncons(kernel, split, []);
        fprintf(fid,'%s\t%s\t%d\t%s\t%.2f\t%.2f\t%0.2f\t%.2f\t%.2f\n', dataset, 'MMD', seed, 'laplace', split.train_bag_prop(1), split.train_bag_prop(2), test_theta,  norm(mmd_props - test_set_proportion', 1), 0);
        
        split.train_bag_prop = original_copy;
        for j = 1:length(training_set_sizes)
            m = split.train_bag_prop(j) * training_set_sizes(j);
            n = (1 - split.train_bag_prop(j)) * training_set_sizes(j);
            
            % Gaussian noise
            noisy_props = gaussian_noise(0.05, 0.05, [m; n]);
            split.train_bag_prop(j) = noisy_props(1);            
        end
        kernel = Ks{mmd_parameters(seed, 1)};
        mmd_props = MMD_uncons(kernel, split, []);
        fprintf(fid,'%s\t%s\t%d\t%s\t%.2f\t%.2f\t%0.2f\t%.2f\t%.2f\n', dataset, 'MMD', seed, 'gaussian', split.train_bag_prop(1), split.train_bag_prop(2), test_theta, norm(mmd_props - test_set_proportion', 1), 0);
        
        split.train_bag_prop = original_copy;
        for j = 1:length(training_set_sizes)
            m = split.train_bag_prop(j) * training_set_sizes(j);
            n = (1 - split.train_bag_prop(j)) * training_set_sizes(j);
            
            % Dirichlet noise
            noisy_props = dirichlet_noise(0.05, 0.05, [m; n]);
            split.train_bag_prop(j) = noisy_props(1);            
        end
        kernel = Ks{mmd_parameters(seed, 1)};
        mmd_props = MMD_uncons(kernel, split, []);
        fprintf(fid,'%s\t%s\t%d\t%s\t%.2f\t%.2f\t%0.2f\t%.2f\t%.2f\n', dataset, 'MMD', seed, 'dirichlet', split.train_bag_prop(1), split.train_bag_prop(2), test_theta,  norm(mmd_props - test_set_proportion', 1), 0);
        
        split.train_bag_prop = original_copy;
        for j = 1:length(training_set_sizes)
            m = split.train_bag_prop(j) * training_set_sizes(j);
            n = (1 - split.train_bag_prop(j)) * training_set_sizes(j);
            
            % Ashwin noise
            noisy_props = dirichlet_noise2(0.05, 0.05, [m; n]);
            split.train_bag_prop(j) = noisy_props(1);            
        end
        kernel = Ks{mmd_parameters(seed, 1)};
        mmd_props = MMD_uncons(kernel, split, []);
        fprintf(fid,'%s\t%s\t%d\t%s\t%.2f\t%.2f\t%0.2f\t%.2f\t%.2f\n', dataset, 'MMD', seed, 'ashwin', split.train_bag_prop(1), split.train_bag_prop(2), test_theta, norm(mmd_props - test_set_proportion', 1), 0);
    end
end
fclose(fid);