function [errors, vars, ys] = plot_error_vs_num_classes()
ys = 2:10;
p = 0.05;
errors = zeros(4, length(ys));
vars = zeros(4, length(ys));
delta = 0.05;
eps = 0.05;

for i = 1:length(ys)
   y = ys(i);
   
   fprintf('%.2f\n', y);
   temp = zeros(3, 20);
   count = 0;
   s = 3000;
   
   for seed = 1:20
       rng(seed);
       pr = [1 - (y-1)*p; ones(y-1, 1)*p];
       count = count + 1;
       
       % Laplace
       props = laplace_noise(eps, delta, floor(pr*s));
       temp(1, count) = norm(props - pr, 1);
       
       % Gaussian
       props = gaussian_noise(eps, delta, floor(pr*s));
       temp(2, count) = norm(props - pr, 1);
       
       % Ashwin
       props = dirichlet_noise2(eps, delta, floor(pr*s));
       temp(3, count) = norm(props - pr, 1);
       
       % Ours
       props = dirichlet_noise(eps, delta, floor(pr*s));
       temp(4, count) = norm(props - pr, 1);
   end
   errors(:, i) = mean(temp, 2);
   vars(:, i) = std(temp, '', 2);
   clear temp;
end

% The new defaults will not take effect if there are any open figures. To
% use them, we close all figures, and then repeat the first example.
close all;

% Defaults for this blog post
width = 3;     % Width in inches
height = 3;    % Height in inches
alw = 0.75;    % AxesLineWidth
fsz = 11;      % Fontsize
lw = 1.5;      % LineWidth
msz = 8;       % MarkerSize

% The properties we've been using in the figures
set(0,'defaultLineLineWidth',lw);   % set the default line width to lw
set(0,'defaultLineMarkerSize',msz); % set the default line marker size to msz
set(0,'defaultLineLineWidth',lw);   % set the default line width to lw
set(0,'defaultLineMarkerSize',msz); % set the default line marker size to msz

% Set the default Size for display
defpos = get(0,'defaultFigurePosition');
set(0,'defaultFigurePosition', [defpos(1) defpos(2) width*100, height*100]);

% Set the defaults for saving/printing to a file
set(0,'defaultFigureInvertHardcopy','on'); % This is the default anyway
set(0,'defaultFigurePaperUnits','inches'); % This is the default anyway
defsize = get(gcf, 'PaperSize');
left = (defsize(1)- width)/2;
bottom = (defsize(2)- height)/2;
defsize = [left, bottom, width, height];
set(0, 'defaultFigurePaperPosition', defsize);

C = {'k','b','r','g','y',[.5 .6 .7],[.8 .2 .6]};
figure
hold on

for i = 1:4
    plot(ys, errors(i, :), 'color', C{i});
end

title(strcat('epsilon = ', num2str(eps), ', delta = ', num2str(delta), ', rho = 0.05'));
xlabel('Number of Classes (c)');
ylabel('Distortion');
axis([2, 10, 0, 0.4]);
legend('Laplace', 'Gaussian', 'Laplace Prior', 'Scaled Dirichlet', 'Location', 'northwest');

end