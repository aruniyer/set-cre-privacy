function run_expt2(dataset)
train_thetas = [0.1:0.1:0.4, 0.49];
test_thetas = 0.1:0.1:0.9;
outfile = strcat(dataset, '_expt2_results');
for j = 1:length(train_thetas)
    train_theta = train_thetas(j);
    expt2(dataset, train_theta, test_thetas, outfile);
end
