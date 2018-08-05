function run_psvm_3_class(dataset, outfile, seeds)

% prefix directory
data_path = strcat('ratio-models/datasets/', dataset);
fprintf(1, 'Datapath = %s\n', data_path);

% create output file
outfile_path = strcat(data_path, '/', outfile);
fid = fopen(outfile_path, 'w+');
fclose(fid);
fprintf(1, 'Outfile = %s\n', outfile);

% load dataset
[~, Y] = dataloader(strcat(data_path, '/full.ssv'));
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
cvparams = load(strcat(data_path, '/psvm_parameters_bak.mat'));
psvm_parameters = cvparams.psvm_parameters_bak;
fprintf(1, 'pSVM parameters loaded.\n');

for i = 1:length(seeds)
    seed = seeds(i);
    split = get_splits(Y, seed, training_set_sizes,...
        training_set_proportions, 10,...
        [0.1,0.1,0.8], 1);
    
    kernel = Ks{psvm_parameters(seed, 1)};
    fprintf(1, 'running pSVM on training data for seed = %d ... ', seed);
    [~, ~, models] = pSVM3(kernel, split, psvm_parameters(2:3));
    fprintf(1, '[DONE]\n');
    test_thetas = 0.1:0.1:0.9;
    for j = 1:length(test_thetas)
        test_theta1 = test_thetas(j);
        for test_theta2 = 0.1:0.1:1-test_theta1
            test_theta3 = 1 - test_theta1 - test_theta2;        
            test_set_proportion = [test_theta1, test_theta2, test_theta3];
            
            fprintf(1, '%.2f, %.2f, %.2f\n', test_theta1, test_theta2, test_theta3);            
            split = get_splits(Y, seed, training_set_sizes,...
                training_set_proportions, testing_set_size,...
                test_set_proportion, 1);
            
            teK = K(split.test_data_idx, split.train_data_idx);
            test_labels = zeros(3, testing_set_size);
            for k = 1:3
                model = models.(strcat('model', num2str(k)));
                test_response =  teK(:, model.support_v) * model.alp + model.b;
                test_labels(k, :) = test_response;
                test_labels(k, test_labels(k,:)>0) = 2^k;
                test_labels(k, test_labels(k,:)<=0) = 0;
            end
            
            msum = sum(test_labels);
            ismem = ismember(msum, [2,4,8]);
            msum(~ismem) = -1;
            msum(msum == 2) = 1;
            msum(msum == 4) = 2;
            msum(msum == 8) = 3;
            if (any(msum == -1))
                ind = find(msum == -1);
                for l = 1:length(ind)
                    [~, indmaxsim] = max(teK(ind(l), :));
                    msum(ind(l)) = split.train_label(indmaxsim);
                end
            end
            
            fid1 = fopen(outfile_path, 'a+');
            fprintf(fid1, '%s\t', dataset);
            fprintf(fid1, '%s\t', method);            
            fprintf(fid1, '%d\t', seed);            
            for k1 = 1:3
                for k2 = 1:3
                    fprintf(fid1, '%0.2f\t', train_bag_prop(k1, k2));
                end
            end
            for k = 1:3
                fprintf(fid1, '%0.2f\t', test_props(k));
            end
            error = sum(abs(psvm_props - test_set_proportion))/3;
            fprintf(fid1, '%0.2f\t', error);
            fprintf(fid1, '%0.2f\t', psvm_acc);
            fprintf(fid1, '\n');
            fclose(fid1);

        end
    end
end

