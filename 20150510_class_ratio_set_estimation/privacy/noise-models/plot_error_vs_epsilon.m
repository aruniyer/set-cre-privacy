function [errors, vars, epss] = plot_error_vs_epsilon()
epss = [0.01:0.01:0.1, 0.2];
errors = zeros(4, length(epss));
vars = zeros(4, length(epss));
delta = 0.05;

for i = 1:length(epss)
   eps = epss(i);
   fprintf('\n%.2f\n', eps);
   temp = zeros(3, 20);
   count = 0;
   s = 1500;
   p = [0.1; 0.1; 0.1; 0.1; 0.6];
   for seed = 1:20;
       count = count + 1;
           
       % Laplace
       props = laplace_noise(eps, delta, p*s);
       temp(1, count) = norm(props - p, 1);
       
       % Gaussian
       props = gaussian_noise(eps, delta, p*s);
       temp(2, count) = norm(props - p, 1);
       
       % Ashwin
       props = dirichlet_noise2(eps, delta, p*s);
       temp(3, count) = norm(props - p, 1);
       
       % Ours
       props = dirichlet_noise(eps, delta, p*s);
       temp(4, count) = norm(props - p, 1);
   end
   errors(:, i) = mean(temp, 2);
   vars(:, i) = std(temp, '', 2);
   clear temp;
end

C = {'k','b','r','g','y',[.5 .6 .7],[.8 .2 .6]};
figure
hold on

% Defaults for this blog post
width = 3;     % Width in inches
height = 3;    % Height in inches
alw = 0.75;    % AxesLineWidth
fsz = 11;      % Fontsize
lw = 1.5;      % LineWidth
msz = 8;       % MarkerSize

pos = get(gcf, 'Position');
set(gcf, 'Position', [pos(1) pos(2) width*100, height*100]); %<- Set size
set(gca, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties

% Set Tick Marks
% set(gca,'XTick',0.01:0.01:0.2);
% set(gca,'YTick',0:0.01:0.1);

for i = 1:4
    errorbar(epss, errors(i, :), vars(i, :), 'color', C{i}, 'LineWidth', lw, 'MarkerSize', msz);
end

title(strcat('delta = ', num2str(delta)));
xlabel('epsilon');
ylabel('Distortion');
axis([0.01, 0.2, 0, 1]);
legend('Laplace', 'Gaussian', 'Laplace Prior', 'Scaled Dirichlet');

end