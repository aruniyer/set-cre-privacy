function [ val ] = delta_func_exp( w, rho, n1, n2, eps )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

    t1 = exp((rho*n1 - 1)*w + (rho*n2 - 1)*log(1-exp(w)) - log(beta(rho*n1, rho*n2)));
    t2 = exp(eps + (rho*n1 - rho - 1)*w + (rho*n2 + rho - 1)*log(1-exp(w)) - log(beta(rho*n1, rho*n2)));
    
    val = t1 - t2;
end
