function [ props, acc, models ] = pSVM3( K, split, parameters )

trK = K(split.train_data_idx, split.train_data_idx);
teK = K(split.test_data_idx, split.train_data_idx);

para.C = parameters(1); % empirical loss weight
para.C_2 = parameters(2); % proportion term weight
para.method = 'alter-pSVM';
para.ep = 0;
para.verbose = 0;

test_labels = zeros(3, length(split.test_label));
acc = 0;

for i = 1:3
    new_split = split;
    new_split.train_label(split.train_label ~= i) = -1;
    new_split.train_label(split.train_label == i) = 1;
    new_split.test_label(split.test_label ~= i) = -1;
    new_split.test_label(split.test_label == i) = 1;
    new_split.train_bag_prop = split.train_bag_prop(:, i);
    new_split.test_bag_prop = split.test_bag_prop(i);
    
    N_random = 20;
    result = [];
    obj = zeros(N_random,1);
    for pp = 1:N_random
        para.init_y = ones(length(trK),1);
        r = randperm(length(trK));
        para.init_y(r(1:floor(length(trK)/2))) = -1;
        result{pp} = test_all_method(new_split, trK, teK, para);
        if (isfield(result{pp}, 'model') && isfield(result{pp}.model,'obj'))
            obj(pp) = result{pp}.model.obj;
        else
            obj(pp) = Inf;
        end
    end
    [~,id] = min(obj);
    result = result{id};
    
    models.(strcat('model', num2str(i))) = result.model;
    acc = acc + result.test_acc;
    test_labels(i, result.predicted_test_label == 1) = 2^i;    
end
msum = sum(test_labels);
ismem = ismember(msum, [2,4,8]);
msum(~ismem) = -1;
msum(msum == 2) = 1;
msum(msum == 4) = 2;
msum(msum == 8) = 3;
if (any(msum == -1))
    ind = find(msum == -1);
    for i = 1:length(ind)
        [~, indmaxsim] = max(teK(ind(i), :));
        msum(ind(i)) = split.train_label(indmaxsim);
    end
end

acc = acc/3;
psvm_props = histc(msum, 1:3)';
props = psvm_props ./ sum(psvm_props);
end