function [errors, vars, ps] = plot_error_vs_proportion()
ps = [10:10:90];
errors = zeros(4, length(ps));
vars = zeros(4, length(ps));
delta = 0.05;
eps = 0.05;

for i = 1:length(ps)
   p = ps(i);
   
   fprintf('%.2f\n', p);
   temp = zeros(3, 20);
   count = 0;
   s = 1000;
   
   %for s = 800:100:2800
       for seed = 1:20
           rng(seed);
           pr = dirrnd([p, 10*ones(1, 4)], 1)';
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
   %end
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

ps = ps / 100;
for i = 1:4
    plot(ps, errors(i, :), 'color', C{i}, 'LineWidth', lw, 'MarkerSize', msz);
end

title(strcat('epsilon = ', num2str(eps), ', delta = ', num2str(delta)));
xlabel('Class Proportion');
ylabel('Distortion');
axis([0.1, 0.9, 0, 0.5]);
% legend('Laplace', 'Gaussian', 'Laplace Prior', 'Scaled Dirichlet');

end