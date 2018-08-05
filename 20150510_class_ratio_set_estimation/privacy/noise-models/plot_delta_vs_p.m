function plot_delta_vs_p(epsilon)
ss = [500, 600, 700, 800, 900, 1000];
ps = 0.1:0.1:0.9;
cs = 0.1:0.1:10;
objs = zeros(length(ss), length(ps));
for j = 1:length(ss)
    s = ss(j);
    for i = 1:length(ps);
        p = ps(i);
        fprintf('%d\t, %.2f\n', s, p);
        vals = zeros(size(cs));
        for k = 1:length(cs)
            alpha = epsilon / cs(k);
            vals(k) = exact_delta_2_class(p, s, epsilon, alpha);
        end        
        objs(j, i) = min(vals);
    end
end
C = {'k','b','r','g','y',[.5 .6 .7],[.8 .2 .6]};
figure
hold on
leg_info = cell(length(ss), 1);

% Defaults for this blog post
width = 3;     % Width in inches
height = 3;    % Height in inches
alw = 1;    % AxesLineWidth
fsz = 14;      % Fontsize
lw = 2;      % LineWidth
msz = 12;       % MarkerSize

pos = get(gcf, 'Position');
set(gcf, 'Position', [pos(1) pos(2) width*100, height*100]); %<- Set size
set(gca, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties

% Set Tick Marks
set(gca,'XTick',0.1:0.1:0.9);
set(gca,'YTick',0:0.01:0.1);

for i = 1:length(ss)
    plot(ps, objs(i, :), 'color', C{i}, 'LineWidth', lw, 'MarkerSize', msz);
    leg_info{i} = ['s = ' num2str(ss(i))];
end
plot(ps, 0.05*ones(size(ps)), 'color', C{7});
title(strcat('eps = ', num2str(epsilon)));
xlabel('p = m/(m+n)');
ylabel('delta');
axis([0.1, 0.9, 0, 0.1]);
legend(leg_info, 'Location', 'northeast');
end

function delta = exact_delta_2_class ( p, s, epsilon, alpha )

fun = @(x) betapdf(x, alpha*p*s, alpha*(1-p)*s);

st = p*exp(epsilon/alpha) / (1 - p + p*exp(epsilon/alpha));
d1 = quadgk(fun, st, 1);

en = 1 - (1-p)*exp(epsilon/alpha) / (p + (1-p)*exp(epsilon/alpha));
d2 = quadgk(fun, 0, en);

delta = max(d1, d2);

end