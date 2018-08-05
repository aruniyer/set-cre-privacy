function check( n1, n2 )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

x = 0:0.01:1;
y = zeros(size(x));
N = 10;
y1 = zeros(N, 1);
y2 = zeros(N, 1);
y3 = zeros(N, 1);
y4 = zeros(N, 1);
y5 = zeros(N, 1);
eps = 0.05;
for n = 1:N
    scaledn1 = n*n1;
    scaledn2 = n*n2;
    y = -delta_func(x, scaledn1, scaledn2, eps);
    y1(n) = max(y);
    
    modepdf1 = max(betapdf(x, scaledn1 - n, scaledn2));
    estimatedmax1 = gamma(scaledn1 + scaledn2) * modepdf1 * gamma(scaledn1 - n)/(gamma(scaledn1 + scaledn2 - n) * gamma(scaledn1));
    y2(n) = estimatedmax1;
    
    estimatedmax2 = (gamma(scaledn1 + scaledn2) * modepdf1 /gamma(scaledn1 + scaledn2 - n)) * (gamma(scaledn1 - n)/gamma(scaledn1) + (n-1)*exp(eps)*gamma(scaledn2)/gamma(scaledn2 + n));
    y3(n) = estimatedmax2;
    
    byalphabeta = gamma(scaledn1 + scaledn2) / (gamma(scaledn1) * gamma(scaledn2));
    estimatedmax3 = byalphabeta * ((scaledn2 - 1)/(scaledn1 + scaledn2))^(scaledn1 + scaledn2);
    y4(n) = estimatedmax3;
    
    modepdf2 = max(betapdf(x, scaledn1, scaledn2));
    y5(n) = modepdf2;
end

y1
y4
y5

end

