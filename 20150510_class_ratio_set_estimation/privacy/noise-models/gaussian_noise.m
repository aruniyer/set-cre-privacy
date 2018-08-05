function [props, ns ] = gaussian_noise( epsilon, delta, sizes )
%GAUSSIAN_NOISE Summary of this function goes here
%   Detailed explanation goes here

s = sum(sizes);
rows = size(sizes, 1);
R = 8*(log (2/delta))/(epsilon*epsilon);
ns = sizes + mvnrnd(zeros(rows, 1), 2*R*eye(rows))';

cvx_quiet(true);
cvx_begin
variable props(size(ns))
minimize(norm(s*props - ns, 1))
subject to
ones(size(ns))'*props == 1
props >= 0
cvx_end

end

