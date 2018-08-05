function [psvm_props_lap, psvm_props_gauss, mmd_props_lap, mmd_props_gauss, psvm_acc_lap, psvm_acc_gauss] = run_psvm_mmd_noisy(datasetname, seed, train_theta, test_theta, epsilon, delta)

[X, Y] = dataloader(strcat('../../../forpedrtest/',datasetname,'_train_',num2str(0.01),'_',num2str(seed),'_',num2str(400)));

training_set_sizes = [100; 100];
training_set_proportions = [train_theta, 1-train_theta; 1-train_theta, train_theta];
test_set_size = 100;
test_set_proportion = [test_theta; 1-test_theta];
split = get_splits( X, Y, seed, training_set_sizes, training_set_proportions, test_set_size, test_set_proportion );

noisy_props = laplace_noise(epsilon, delta, [label1, label2]');
split.train_bag_prop = [noisy_props(1); 1 - noisy_props(1)];

kernel_type = 2;
psvm_parameters{1} = kernel_type;
psvm_parameters{2} = 1;
mmd_parameters{1} = kernel_type;

[psvm_props_lap, psvm_acc_lap] = pSVM( X, split, psvm_parameters );
mmd_props_lap = MMD( X, split, mmd_parameters );

noisy_props = gaussian_noise(epsilon, delta, [label1, label2]');
split.train_bag_prop = [noisy_props(1); 1 - noisy_props(1)];

[psvm_props_gauss, psvm_acc_gauss] = pSVM( X, split, psvm_parameters );
mmd_props_gauss = MMD( X, split, mmd_parameters );

end
