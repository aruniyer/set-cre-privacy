function [ K ] = kernel_f(data, method)
%KERNEL_F Summary of this function goes here
%   Detailed explanation goes here
switch(method)
    case 1
        K = data*data';
    case 2
        K = pdist2(data, data);
        bw = 1/8;
        K = exp(-K/2/bw^2);
    case 3
        K = pdist2(data, data);
        bw = 1/4;
        K = exp(-K/2/bw^2);
    case 4
        K = pdist2(data, data);
        K = exp(-1*K);
    case 5
        K = pdist2(data, data);
        bw = 1;
        K = exp(-K/2/bw^2);
    case 6
        K = pdist2(data, data);
        K = exp(-0.1*K);
    case 7
        K = pdist2(data, data);
        bw = 4;
        K = exp(-K/2/bw^2);
    case 8
        K = pdist2(data, data);
        K = exp(-0.01*K);
end
end