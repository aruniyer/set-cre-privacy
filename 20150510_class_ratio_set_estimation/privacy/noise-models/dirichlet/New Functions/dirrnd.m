function r = dirrnd(alpha, n)
% take a sample from a dirichlet distribution
p = length(alpha);
r = gamrnd(repmat(alpha,n,1),1,n,p);
r = r ./ repmat(sum(r,2),1,p);