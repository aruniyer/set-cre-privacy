function [ props, ns ] = dirichlet_noise2( epsilon, delta, sizes)
%LAPLACE_NOISE Summary of this function goes here
%   Detailed explanation goes here

ns = sizes + laprnd(size(sizes, 1), size(sizes, 2), 0, 2 /epsilon);
ns(ns < 0) = 0;
props = dirrnd(ns', 1)';

end

