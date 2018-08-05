function run_mmd_3_class(dataset, outfile)

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
fprintf(1, 'Params loaded.\n');

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
            
            kernel = Ks{mmd_parameters(seed, 1)};
            mmd_props = MMD(kernel, split, []);
            fprintf(fid, '%s\t', dataset);
            fprintf(fid, '%s\t', 'MMD');
            fprintf(fid, '%d\t', seed);
            for i = 1:3
                for j = 1:3
                    fprintf(fid, '%0.2f\t', split.train_bag_prop(i, j));
                end
            end
            fprintf(fid, '%0.2f\t', test_theta1);
            fprintf(fid, '%0.2f\t', test_theta2);
            fprintf(fid, '%0.2f\t', test_theta3);
            error = sum(abs(mmd_props - [test_theta1;test_theta2;test_theta3]))/3;
            fprintf(fid, '%0.2f\t', error);
            fprintf(fid, '%0.2f\t', 0);
            fprintf(fid, '\n');
        end
    end
end
fclose(fid);