function train_test_creator(data_path)
[X, Y] = dataloader(strcat(data_path, '/full.ssv'));

Ks = cell(3, 1);
K = load(strcat(data_path, '/kernel_gamma_1.mat'));
Ks{1} = K.K1;
K = load(strcat(data_path, '/kernel_gamma_0p1.mat'));
Ks{2} = K.K1;
K = load(strcat(data_path, '/kernel_gamma_0p01.mat'));
Ks{3} = K.K1;

for i = 1:3
    size(Ks{i});
end

training_set_sizes = [...
    600, 600, 600, 600,...
];
training_set_proportions = [...
    0.1, 0.9;...
    0.9, 0.1;...
    0.1, 0.9;...
    0.9, 0.1;...
];
    
testing_set_size = 10;
mmd_parameters = zeros(10, 1);
for seed = 1:10
    test_set_proportion = [0.5, 0.5];
    split = get_splits(Y, seed, training_set_sizes,...
        training_set_proportions, testing_set_size,...
        test_set_proportion);
        
    bws = [1,2,3];
    empty = [0];
    parameters = cross_validate(Ks, split, @MMD, {bws; empty; empty});
    mmd_parameters(seed) = parameters(1);    
end
save(strcat(data_path, '/mmd_parameters.mat'), 'mmd_parameters');

psvm_parameters = zeros(10, 3);
for seed = 1:10
    test_set_proportion = [0.5, 0.5];
    split = get_splits(Y, seed, training_set_sizes,...
        training_set_proportions, testing_set_size,...
        test_set_proportion);
        
    bws = [1,2,3];
    Cs = [0.1, 1, 10];
    Cps = [1, 10, 100];        
    parameters = cross_validate(Ks, split, @pSVM, {bws; Cs; Cps});
    psvm_parameters(seed, :) = parameters;    
end
save(strcat(data_path, '/psvm_parameters.mat'), 'psvm_parameters');

end