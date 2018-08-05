function [ val ] = delta_func( x, n1, n2, eps )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

t1 = x.^(n1 - 1).*(1 - x).^(n2 - 1)./Beta(n1, n2);
t2 = x.^(n1 - 2).*(1 - x).^(n2)./Beta(n1 - 1, n2 + 1);


val = exp(eps) * t2 - t1;

end

function [z] = Beta(n1, n2)
z = gamma(n1) * gamma (n2) / gamma(n1 + n2);
end