function [ mmd_prop_mat, psvm_prop_mat, psvm_acc_mat ] = run_psvm_mmd(datasetname, seed, train_theta, test_thetas)

[X, Y] = dataloader(strcat('../../../forpedrtest/',datasetname,'_train_',num2str(0.01),'_',num2str(seed),'_',num2str(400)));
[X1, Y1] = dataloader(strcat('../../../forpedrtest/',datasetname,'_train_',num2str(0.1),'_',num2str(seed),'_',num2str(400)));
X = [X; X1];
Y = [Y; Y1];
clear X1 Y1;

training_set_sizes = [100; 100; 100];
training_set_proportions = [train_theta, 1-train_theta; 1-train_theta, train_theta; 0.5, 0.5];
test_set_size = 100;

mmd_prop_mat = zeros(length(test_thetas), 2);
psvm_prop_mat = zeros(length(test_thetas), 2);
psvm_acc_mat = zeros(length(test_thetas), 1);
for i = 1:length(test_thetas)
    test_theta = test_thetas(i);
    test_set_proportion = [test_theta; 1-test_theta];
    split = get_splits( Y, seed, training_set_sizes, training_set_proportions, test_set_size, test_set_proportion );
    
    if i == 1
        bws = [4,6,8];
        Cs = [0.1, 1, 10];
        Cps = [1, 10, 100];        
        
        winner = cross_validate( X, split, @MMD, {bws} );
        mmd_kernel_type = winner(1);
        
        winner = cross_validate( X, split, @pSVM, {bws; Cs; Cps} );
        psvm_kernel_type = winner(1);
        psvm_empirical_weight = winner(2);
        psvm_proportion_weight = winner(3);
    end
    
    psvm_parameters = [psvm_kernel_type, psvm_empirical_weight, psvm_proportion_weight];
    [psvm_props, psvm_acc] = pSVM( X, split, psvm_parameters );
    
    mmd_parameters(1) = mmd_kernel_type;
    mmd_props = MMD( X, split, mmd_parameters );
    
    mmd_prop_mat(i, :) = mmd_props;
    psvm_prop_mat(i, :) = psvm_props;
    psvm_acc_mat(i) = psvm_acc;
end

end
