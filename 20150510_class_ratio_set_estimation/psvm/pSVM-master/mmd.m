function [alpha, obj] = mmd(K, S, U)

M = size(S, 1);

H = zeros(M);
f = zeros(M, 1);
for i = 1:M
    idx1 = S(i, :);
    for j = 1:M
        idx2 = S(j, :);
        H(i, j) = 2*mean(mean(K(idx1, idx2)));
    end    
    f(i) = -2 * mean(mean(K(U, idx1)));
end

H = H + eye(size(H));

A = [];
b = [];
Aeq = ones(1, M);
beq = 1;
lb = [];
ub = [];
alpha0 = ones(1, M) / M;

options = optimset('Algorithm', 'active-set', 'Display', 'Off', 'LargeScale', 'Off');

alpha = quadprog(H, f, A, b, Aeq, beq, lb, ub, alpha0, options);

obj = .5 * (alpha'*H*alpha) + alpha'*f;