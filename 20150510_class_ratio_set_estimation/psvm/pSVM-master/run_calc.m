f = 0.1:0.1:5;
epss = [0, 0.05, 0.1, 0.2];
colors = hsv(5);
for v = 0.01:0.01:0.25
    figure
    hold on
    for eps_i = 1:length(epss)
        eps = epss(eps_i);
        y = epsdelta_in_r(f, v, eps);
        plot(f, y, 'Color', colors(eps_i, :))
    end
    legend('eps = 0', 'eps = 0.05', 'eps = 0.1', 'eps = 0.2')
    xlabel('m/n')
    ylabel('delta')
    title(strcat('v = ', num2str(v)))
    axis([0, 5, 0, 1])
    hold off
end
