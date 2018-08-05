function [X, Y] = dataloader(datasetpath, fixy)

A = load(datasetpath);
nf = size(A, 2);
X = A(:, 1:nf - 1);
Y = A(:,nf);

% change labels from 0..c to 1..c+1
if (~exist('fixy', 'var') || fixy == 1)
    nc = length(unique(Y));
    for i=nc-1:-1:0
        idx = Y == i;
        Y(idx) = (i + 1);
    end
end
