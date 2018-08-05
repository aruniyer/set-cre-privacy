function [ props, ns ] = laplace_noise( epsilon, delta, sizes)
%LAPLACE_NOISE Summary of this function goes here
%   Detailed explanation goes here

s = sum(sizes);
ns = sizes + laprnd(size(sizes, 1), size(sizes, 2), 0, 2 /epsilon);

cvx_quiet(true);
cvx_begin
variable props(size(ns))
minimize(norm(s*props - ns, 1))
subject to
ones(size(ns))'*props == 1
props >= 0
cvx_end

end

