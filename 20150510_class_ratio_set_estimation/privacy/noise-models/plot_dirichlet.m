function plot_dirichlet(epsilon, c_or_p, plotname)

switch plotname
    case 'dvc' 
        plot_dirichlet_delta_vs_c(epsilon, c_or_p);
    case 'dvp' 
        plot_dirichlet_delta_vs_p(epsilon, c_or_p);
end

end

function plot_dirichlet_delta_vs_c(epsilon, p)
ss = 600:100:600;
cs = [0.1:0.1:1, 2:10];
objs1 = zeros(length(ss), length(cs));
objs2 = zeros(length(ss), length(cs));
for j = 1:length(ss)
    s = ss(j);
    for i = 1:length(cs);        
        c = cs(i);
        fprintf('%d\t%.2f\n', s, c);
        alpha = epsilon / c;
%         objs1(j, i) = exact_delta_2_class(p, s, epsilon, alpha);
        objs2(j, i) = exact_delta_full(s*p, epsilon, alpha);
    end
    fprintf('%d\t%.4f\t%.4f\n', ss(j), min(objs1(j, :)),min(objs2(j, :)));
end
C = {'k','b','r','g','y',[.5 .6 .7],[.8 .2 .6]};
figure
hold on
leg_info = cell(length(ss), 1);
for i = 1:length(ss)
    plot(cs, objs1(i, :), C{i+1}, cs, objs2(i, :)); %, 'color', C{i});
    leg_info{i} = ['s = ' num2str(ss(i))];
end
title(strcat('eps = ', num2str(epsilon), ', p = ', num2str(p)));
xlabel('c = epsilon/alpha');
ylabel('exact delta');
axis([0, 10, 0, 1]);
legend(leg_info, 'Location', 'northwest');
end

function plot_dirichlet_delta_vs_p(epsilon, c)
ss = [30, 40, 50, 70, 100, 1000, 10000];
ps = 0.1:0.01:0.9;
objs = zeros(length(ss), length(ps));
alpha = epsilon / c;
for j = 1:length(ss)
    s = ss(j);
    for i = 1:length(ps);
        p = ps(i);
        objs(j, i) = exact_delta(p, s, epsilon, alpha);
    end
end
C = {'k','b','r','g','y',[.5 .6 .7],[.8 .2 .6]};
figure
hold on
leg_info = cell(length(ss), 1);
for i = 1:length(ss)
    plot(ps, objs(i, :), 'color', C{i});
    leg_info{i} = ['s = ' num2str(ss(i))];
end
title(strcat('eps = ', num2str(epsilon), ', c = ', num2str(c), ', alpha = ', num2str(alpha)));
xlabel('p = m/(m+n)');
ylabel('exact delta');
axis([0, 1, 0, 1]);
legend(leg_info, 'Location', 'northwest');
end

function pdf = dummy(x, y, fact, fun)
% x, y, fun(x, y)
pdf  = (x > fact*y).*fun(x, y);
pdf(isnan(pdf)) = 0;
end

function delta = exact_delta_full ( sizes, epsilon, alpha )

m = sum(sizes);

max_delta = 0;
for i = 1:length(sizes)
    for j = 1:length(sizes)
        if (i == j) 
            continue;
        end
        eta1 = sizes(i);
        eta2 = sizes(j);
        dirpar = [alpha*eta1, alpha*eta2, alpha*(m - eta1 - eta2)];
        fun = @(x, y) dirpdf([x', (y*ones(size(x)))'], dirpar);
        lambda12 = exp(gammaln(alpha*eta1) + gammaln(alpha*eta2) - gammaln(alpha*eta1 - alpha) - gammaln(alpha*eta2 + alpha));
        fact = lambda12^(1/alpha)*exp(epsilon/alpha);
        curr_delta = dblquad(@(x,y) dummy(x,y,fact,fun), 0, 1, 0, 1);
%         if (fact < 10)
%             [fact, curr_delta];
%         end
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