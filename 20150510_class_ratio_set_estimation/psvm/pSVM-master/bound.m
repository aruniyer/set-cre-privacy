size = 64000;
ns = 8;
ortho = sqrt(ns);
maxR = 1;
maxS = maxR;
delta = 10^-5:(0.3 - 10^-5)/100:0.3;
y1 = zeros(length(delta), 1);
for i = 1:length(delta)
    y1(i) = compute_bound(ortho, size, ns, maxR, maxS, delta(i)); 
end
M = [delta' y1]; 
% csvwrite('delta.csv', M);
figure,
plot(delta, y1), 
legend('error by delta'), 
title('size = 64000, num_sets = 8'), 
ylabel('error bound'), 
xlabel('delta'),
ylim(gca, [0 1]),
set(gca, 'fontsize', 28, 'fontweight','bold'),
set(findall(gcf, 'type', 'text'), 'fontsize', 28, 'fontweight', 'bold');

size = 64000;
ns = 2:16;
ortho = sqrt(ns);
maxR = 1;
maxS = maxR;
delta = 10^-5;
y2 = zeros(length(ns), 1);
for i = 1:length(ns)
    y2(i) = compute_bound(ortho(i), size, ns(i), maxR, maxS, delta); 
end
M = [ns' y2]; 
% csvwrite('delta.csv', M);
figure,
plot(ns', y2), 
legend('error by num sets'), 
title('size = 64000, delta = 10^-5'), 
ylabel('error bound'), 
xlabel('num sets'),
ylim(gca, [0 1]),
set(gca, 'fontsize', 28, 'fontweight','bold'),
set(findall(gcf, 'type', 'text'), 'fontsize', 28, 'fontweight', 'bold');

size = 16000:10000:96000;
ns = 8;
ortho = sqrt(ns);
maxR = 1;
maxS = maxR;
delta = 10^-5;
y3 = zeros(length(size), 1);
for i = 1:length(size)
    y3(i) = compute_bound(ortho, size(i), ns, maxR, maxS, delta); 
end
M = [size' y3]; 
% csvwrite('delta.csv', M);
figure,
plot(size', y3), 
legend('error by size'), 
title('num sets = 8, delta = 10^-5'), 
ylabel('error bound'), 
xlabel('set size'),
ylim(gca, [0 1]),
set(gca, 'fontsize', 28, 'fontweight','bold'),
set(findall(gcf, 'type', 'text'), 'fontsize', 28, 'fontweight', 'bold');

size = 64000;
ns = 8;
ortho = sqrt(ns);
maxR = 1;
sbyr = 0.1:0.1:2;
maxS = maxR * sbyr;
delta = 10^-5;
y4 = zeros(length(sbyr), 1);
for i = 1:length(sbyr)
    y4(i) = compute_bound(ortho, size, ns, maxR, maxS(i), delta);
end
M = [sbyr' y4];
% csvwrite('delta.csv', M);
figure,
plot(sbyr', y4), 
legend('error by S/R'), 
title('set size = 64000, num sets = 8, delta = 10^-5'), 
ylabel('error bound'), 
xlabel('S/R'),
ylim(gca, [0 1]),
set(gca, 'fontsize', 28, 'fontweight','bold'),
set(findall(gcf, 'type', 'text'), 'fontsize', 28, 'fontweight', 'bold');