function run_mmd_3_class_noisy(dataset, outfile)

% prefix directory
data_path = strcat('ratio-models/datasets/', dataset);
fprintf('Data path = %s\n', data_path);

% create output file
outfile_path = strcat(data_path, '/', outfile);
fid = fopen(outfile_path, 'w+');
fprintf('Outfile = %s\n', outfile_path);

% load dataset
[X, Y] = dataloader(strcat(data_path, '/full.ssv'));
fprintf('Data loaded.\n');

% load pre-computed kernels
Ks = cell(3, 1);
K = load(strcat(data_path, '/kernel_gamma_1.mat'));
Ks{1} = K.K1;
K = load(strcat(data_path, '/kernel_gamma_0p1.mat'));
Ks{2} = K.K1;
K = load(strcat(data_path, '/kernel_gamma_0p01.mat'));
Ks{3} = K.K1;
fprintf('Kernels loaded.\n');

% set train sizes and proportions
training_set_sizes = [...
    900, 900, 900, 900, 900,...
];
training_set_proportions = [...
    0.1, 0.1, 0.8;...
    0.1, 0.8, 0.1;...
    0.8, 0.1, 0.1;...
    0.1, 0.1, 0.8;...
    0.1, 0.8, 0.1;...    
];
    
% set test set size
testing_set_size = 900;

% load cross-validated parameters
cvparams = load(strcat(data_path, '/mmd_parameters'));
mmd_parameters = cvparams.mmd_parameters;
fprintf('Params loaded.\n');

% for every seed, every proportion run mmd
for seed = 1:10
    for test_theta1 = 0.1:0.1:0.9
        for test_theta2 = 0.1:0.1:1-test_theta1
            test_theta3 = 1 - test_theta1 - test_theta2;
            fprintf(1, '%d, %.2f, %.2f, %.2f\n', seed, test_theta1, test_theta2, test_theta3);
            test_set_proportion = [test_theta1, test_theta2, test_theta3];
            split = get_splits(Y, seed, training_set_sizes,...
                training_set_proportions, testing_set_size,...
                test_set_proportion, 1);
            original_copy = split.train_bag_prop;
            
            for j = 1:length(training_set_sizes)
                % Laplace noise
                noisy_props = laplace_noise(0.05, 0.05, training_set_sizes(j)*training_set_proportions(j, :)');
                split.train_bag_prop(j, :) = noisy_props';            
            end
            kernel = Ks{mmd_parameters(seed, 1)};
            mmd_props = MMD(kernel, split, []);
            test_props = [test_theta1;test_theta2;test_theta3];
            printToFile(fid, dataset, 'MMD', seed, 'laplace', split.train_bag_prop, test_props, mmd_props);
            
            split.train_bag_prop = original_copy;
            for j = 1:length(training_set_sizes)
                % Gaussian noise
                noisy_props = gaussian_noise(0.05, 0.05, training_set_sizes(j)*training_set_proportions(j, :)');
                split.train_bag_prop(j, :) = noisy_props';            
            end
            kernel = Ks{mmd_parameters(seed, 1)};
            mmd_props = MMD(kernel, split, []);
            test_props = [test_theta1;test_theta2;test_theta3];
            printToFile(fid, dataset, 'MMD', seed, 'gaussian', split.train_bag_prop, test_props, mmd_props);
            
            split.train_bag_prop = original_copy;
            for j = 1:length(training_set_sizes)
                % Dirichlet noise
                noisy_props = dirichlet_noise(0.05, 0.05, training_set_sizes(j)*training_set_proportions(j, :)');
                split.train_bag_prop(j, :) = noisy_props';            
            end
            kernel = Ks{mmd_parameters(seed, 1)};
            mmd_props = MMD(kernel, split, []);
            test_props = [test_theta1;test_theta2;test_theta3];
            printToFile(fid, dataset, 'MMD', seed, 'dirichlet', split.train_bag_prop, test_props, mmd_props);
            
            split.train_bag_prop = original_copy;
            for j = 1:length(training_set_sizes)
                % Ashwin noise
                noisy_props = dirichlet_noise2(0.05, 0.05, training_set_sizes(j)*training_set_proportions(j, :)');
                split.train_bag_prop(j, :) = noisy_props';            
            end
            kernel = Ks{mmd_parameters(seed, 1)};
            mmd_props = MMD(kernel, split, []);
            test_props = [test_theta1;test_theta2;test_theta3];
            printToFile(fid, dataset, 'MMD', seed, 'ashwin', split.train_bag_prop, test_props, mmd_props);
        end
    end
end
fclose(fid);
end

function printToFile(fid, dataset, method, seed, noise, train_bag_prop, test_props, mmd_props)
fprintf(fid, '%s\t', dataset);
fprintf(fid, '%s\t', method);            
fprintf(fid, '%d\t', seed);            
fprintf(fid, '%s\t', noise);            
for i = 1:3
    for j = 1:3
        fprintf(fid, '%0.2f\t', train_bag_prop(i, j));
    end
end
for i = 1:3
    fprintf(fid, '%0.2f\t', test_props(i));
end
error = norm(mmd_props - test_props, 1);
fprintf(fid, '%0.2f\t', error);
fprintf(fid, '%0.2f\t', 0);
fprintf(fid, '\n');
end