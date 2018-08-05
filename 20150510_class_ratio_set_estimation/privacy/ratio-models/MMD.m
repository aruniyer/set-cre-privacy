function [ props, alpha ] = MMD( K, split, parameters )

M = length(split.train_bag_prop);
H = zeros(M);
f = zeros(M, 1);
for i = 1:M
    idx1 = split.train_data_idx(split.train_bag_idx == i);
    for j = 1:M
        idx2 = split.train_data_idx(split.train_bag_idx == j);
        H(i, j) = 2*mean(mean(K(idx1, idx2)));
    end    
    f(i) = -2 * mean(mean(K(split.test_data_idx, idx1)));
end
H = (H + H')/2;

A = [];
b = [];
Aeq = ones(1, M);
beq = 1;
lb = [];
ub = [];
alpha0 = ones(M, 1);

options = optimset('Algorithm', 'active-set', 'Display', 'Off', 'LargeScale', 'Off');
alpha = quadprog(H, f, A, b, Aeq, beq, lb, ub, alpha0, options);
if (size(split.train_bag_prop, 2) == 1)
    % if binary case, handle the format used by psvm
    p = alpha' * split.train_bag_prop;
    props = [p; 1-p];
else
    props = alpha' * split.train_bag_prop;
    props = props';
end