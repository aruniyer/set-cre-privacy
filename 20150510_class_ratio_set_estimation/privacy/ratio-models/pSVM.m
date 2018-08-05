function [ props, acc ] = pSVM( K, split, parameters )

trK = K(split.train_data_idx, split.train_data_idx);
teK = K(split.test_data_idx, split.train_data_idx);

para.C = parameters(1); % empirical loss weight
para.C_2 = parameters(2); % proportion term weight
para.method = 'alter-pSVM';
para.ep = 0;
para.verbose = 0;

N_random = 20;
result = [];
obj = zeros(N_random,1);
for pp = 1:N_random
    para.init_y = ones(length(trK),1);
    r = randperm(length(trK));
    para.init_y(r(1:floor(length(trK)/2))) = -1;
    result{pp} = test_all_method(split, trK, teK, para);
    if (isfield(result{pp}, 'model') && isfield(result{pp}.model,'obj'))
        obj(pp) = result{pp}.model.obj;
    else
        obj(pp) = Inf;
    end
end
[mm,id] = min(obj);
result = result{id};

acc = result.test_acc;
psvm_res = histc(result.predicted_test_label, -1:1);
psvm_props = [psvm_res(3), psvm_res(1)]';
props = psvm_props ./ sum(psvm_props);

end