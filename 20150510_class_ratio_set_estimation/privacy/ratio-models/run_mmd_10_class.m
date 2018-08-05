function run_mmd_10_class(dataset, outfile)

% prefix directory
data_path = strcat('ratio-models/datasets/', dataset);
fprintf(1, 'Datapath = %s\n', data_path);

% create output file
outfile_path = strcat(data_path, '/', outfile);
fid = fopen(outfile_path, 'w+');
fprintf(1, 'Outfile = %s\n', outfile_path);

% load dataset
[X, Y] = dataloader(strcat(data_path, '/full.ssv'));
fprintf(1, 'Data loaded\n');

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
    900, 900, 900, 900, 900, 900, 900, 900, 900, 900,...
];
training_set_proportions = [...
    0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.55;...
    0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.55, 0.05;...
    0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.55, 0.05, 0.05;...
    0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.55, 0.05, 0.05, 0.05;...
    0.05, 0.05, 0.05, 0.05, 0.05, 0.55, 0.05, 0.05, 0.05, 0.05;...
    0.05, 0.05, 0.05, 0.05, 0.55, 0.05, 0.05, 0.05, 0.05, 0.05;...
    0.05, 0.05, 0.05, 0.55, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05;...
    0.05, 0.05, 0.55, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05;...
    0.05, 0.55, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05;...
    0.55, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05;...
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
        para_prior = [test_theta1, (1-test_theta1)/9*ones(1, 9)];
        alphas = 500*para_prior;
        for seedi = 1:1
            rng(seed);
            test_set_proportion = dirrnd(alphas, 1);
            fprintf(1, '%d, ', seed);
            for k = 1:10
                fprintf(1, '%.2f, ', test_set_proportion(k));
            end            
            fprintf(1, '\n');
            
            split = get_splits(Y, seed, training_set_sizes,...
                training_set_proportions, testing_set_size,...
                test_set_proportion, 1);
            kernel = Ks{mmd_parameters(seed, 1)};
            mmd_props = MMD(kernel, split, []);
            printToFile(fid, dataset, 'MMD', seed, 'None', split.train_bag_prop, test_set_proportion', mmd_props);
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
    fprintf(fid, '%0.2f\t', train_bag_prop(i, 1));
end
fprintf(fid, '%0.2f\t', test_props(1));
fprintf(fid, '%0.2f\t', roundn(test_props(1), -1));
error = norm(mmd_props - test_props, 1);
fprintf(fid, '%0.2f\t', error);
fprintf(fid, '%0.2f\t', 0);
fprintf(fid, '\n');
end