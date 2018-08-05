function [ split ] = get_splits( Y, seed, training_set_sizes,...
    training_set_proportions, test_set_size, test_set_proportions, multi )
% Y - labels (unique labels |Y| = c)
% training_set_sizes (M x 1) - sizes of required sets
% training_set_proportions (M x c) - proportions for each set
% test_set_size (1 x 1) - size of test set
% test_set_proportions (1 x c) - proportion of test set

M = length(training_set_sizes);
c = length(unique(Y));
ny = histc(Y, 1:c);

for y = 1:c
    nyreq = 0;
    for sett = 1:M
        nyreq = nyreq + training_set_sizes(sett) *...
            training_set_proportions(sett, y);
    end
    if (nyreq > ny(y))
        error(strcat('Cannot create desired split. y = ', num2str(y),...
            ', nyreq = ', num2str(nyreq), ', ny(y) = ', num2str(ny(y))));
    end
end

split.train_data_idx = zeros(sum(training_set_sizes), 1);
rng(seed);
j = 0;
m = 0;
for y = 1:c
    ids_for_label_y = find(Y == y);
    p = randperm(ny(y));
    k = 0;
    for sett = 1:M
        ny_set = floor(training_set_sizes(sett) * training_set_proportions(sett, y));
        split.train_data_idx(j + 1:j + ny_set) = ids_for_label_y(p(k + 1:k + ny_set));
        split.train_bag_idx(j + 1:j + ny_set) = sett;
        k = k + ny_set;
        j = j + ny_set;
    end
    p = randperm(ny(y));
    ny_test = floor(test_set_size * test_set_proportions(y));
    split.test_data_idx(m + 1:m + ny_test) = ids_for_label_y(p(1:ny_test));
    split.test_bag_idx(m + 1:m + ny_test) = 1;
    m = m + ny_test;
end

split.train_label = Y(split.train_data_idx);
split.test_label = Y(split.test_data_idx);

if (~exist('multi', 'var'))
    % the following piece is written with binary class in mind
    split.train_bag_prop = training_set_proportions(:, 1);
    split.test_bag_prop = test_set_proportions(1);
    split.train_label(split.train_label == 2) = -1;
    split.test_label(split.test_label == 2) = -1;
elseif (multi == 1)
    split.train_bag_prop = training_set_proportions;
    split.test_bag_prop = test_set_proportions;    
end

end