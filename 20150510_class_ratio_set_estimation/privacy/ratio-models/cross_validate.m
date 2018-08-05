function [ winner, min_error ] = cross_validate( Ks, split, model, parameters_list )
%CROSS_VALIDATE Summary of this function goes here
%   Detailed explanation goes here

cart_prod = cartesian_product(parameters_list);
rows = size(cart_prod, 1);
errors = zeros(rows, 1);
for index = 1:rows 
    parameters = cart_prod(index, :);
    fprintf(1, '%.2f\t', parameters);
    fprintf(1, '\n');
    error = leave_one_out(Ks{parameters(1)}, split, model, parameters(2:3));
    errors(index) = error;
end
errors
[min_error, win_index] = min(errors);
winner = cart_prod(win_index, :);
end

function [ average_error ] = leave_one_out ( K, split, model, parameters )
M = length(split.train_bag_prop);

average_error = 0;
for i = 1:M
    new_split.train_data_idx = split.train_data_idx(split.train_bag_idx ~= i);
    new_split.train_bag_idx = split.train_bag_idx(split.train_bag_idx ~= i);
    bag_ids = unique(new_split.train_bag_idx);
    for j = 1:M-1
        new_split.train_bag_idx(new_split.train_bag_idx == bag_ids(j)) = j;
    end
    new_split.train_bag_prop = split.train_bag_prop(bag_ids);
    new_split.train_label = split.train_label(split.train_bag_idx ~= i);
    new_split.test_data_idx = split.train_data_idx(split.train_bag_idx == i);
    new_split.test_bag_idx = split.train_bag_idx(split.train_bag_idx == i);
    new_split.test_bag_idx(new_split.test_bag_idx == i) = 1;
    new_split.test_bag_prop = split.train_bag_prop(i);
    new_split.test_label = split.train_label(split.train_bag_idx == i);
    
    props = model( K, new_split, parameters );
    error = abs(props(1) - split.test_bag_prop);
    average_error = average_error + error;
end
average_error = average_error / M;

end

