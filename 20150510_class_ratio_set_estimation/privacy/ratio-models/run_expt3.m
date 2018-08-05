function run_expt3(dataset)
train_thetas = [0.1:0.1:0.4, 0.49];
test_thetas = 0.1:0.1:0.9;
outfile = strcat(dataset, '_expt3_results');
for j = 1:length(train_thetas)
    for k = 1:length(test_thetas)
        train_theta = train_thetas(j);
        test_theta = test_thetas(k);
        expt3(dataset, train_theta, test_theta, outfile);
    end
end