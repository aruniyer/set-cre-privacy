
lambda12 = exp(gammaln(alpha*eta1) + gammaln(alpha*eta2) - gammaln(alpha*eta1 - alpha) - gammaln(alpha*eta2 + alpha));
p = [0.1, 0.1, 0.1, 0.1, 0.6];
for i = 1:5
    for j = 1:5
        if i == j
            continue;
        end
        p(i)*(1-p(i)) + 
    end
end