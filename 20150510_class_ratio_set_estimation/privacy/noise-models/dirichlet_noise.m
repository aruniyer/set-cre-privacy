function [ props, ns ] = dirichlet_noise( epsilon, delta, sizes )
%DIRICHLET Summary of this function goes here
%   Detailed explanation goes here

if (length(sizes) == 2)
    s = sum(sizes);
    p = sizes(1)/s;
    cs = 0.1:0.1:10;
    objs = zeros(length(cs));
    for i = length(cs)
        c = cs(i);
        alpha = epsilon / c;
        objs(i) = exact_delta_2_class(p, s, epsilon, alpha);
    end
    [val, ind] = min(objs);
    
    if val(1) < delta    
        alpha = epsilon / cs(ind(1));        
        pr = betarnd(floor(alpha*p*s), floor(alpha*(1-p)*s));
        if (isnan(pr))
            pr = p;
        end
        props = [pr; 1-pr];
        ns = sizes;
    else
        error(strcat('unable to satisfy min delta for eps = ', num2str(epsilon), ', delta = ', num2str(delta), ', m = ', sizes(1), ', n = ', sizes(2)));
    end
else
    betas = [0.1:0.1:1,2,3];
    objs = zeros(length(betas));
    for i = length(betas)
        beta = betas(i);
        alpha = epsilon / beta;
        objs(i) = exact_delta_full(sizes, epsilon, alpha);
    end
    [val, ind] = min(objs);
    
    if val(1) < delta    
        alpha = epsilon / betas(ind(1));
        props = dirrnd(alpha*sizes', 1)';
        if (any(isnan(props)))
            error('NaN entries');
            props = sizes/sum(sizes);
        end
        ns = sizes;        
    else
        error(strcat('unable to satisfy min delta for eps = ', num2str(epsilon), ', delta = ', num2str(delta)));
    end
end

end

function pdf = dirichlet_pdf(x, y, fact, fun)
pdf  = (x > fact*y).*fun(x, y);
pdf(isnan(pdf)) = 0;
end

function delta = exact_delta_full ( sizes, epsilon, alpha )

m = sum(sizes);
[~, ind] = sort(sizes);

max_delta = 0;
for i = ind(1):ind(1)
    for j = ind(2):ind(2)
        if (i == j) 
            continue;
        end
        eta1 = sizes(i);
        eta2 = sizes(j);
        dirpar = [alpha*eta1, alpha*eta2, alpha*(m - eta1 - eta2)];
        fun = @(x, y) dirpdf([x', (y*ones(size(x)))'], dirpar);
        lambda12 = exp(gammaln(alpha*eta1) + gammaln(alpha*eta2) - gammaln(alpha*eta1 - alpha) - gammaln(alpha*eta2 + alpha));
        fact = lambda12^(1/alpha)*exp(epsilon/alpha);
        curr_delta = dblquad(@(x, y) dirichlet_pdf(x, y, fact, fun), 0, 1, 0, 1);
        if curr_delta > max_delta
            max_delta = curr_delta;
        end
    end
end

delta = max_delta;
end

function delta = exact_delta_2_class ( p, s, epsilon, alpha )

fun = @(x) betapdf(x, alpha*p*s, alpha*(1-p)*s);

st = p*exp(epsilon/alpha) / (1 - p + p*exp(epsilon/alpha));
d1 = quadgk(fun, st, 1);

en = 1 - (1-p)*exp(epsilon/alpha) / (p + (1-p)*exp(epsilon/alpha));
d2 = quadgk(fun, 0, en);

delta = max(d1, d2);

end