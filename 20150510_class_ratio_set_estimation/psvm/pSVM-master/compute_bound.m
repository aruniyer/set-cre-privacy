function [bnd] = compute_bound(ortho, set_size, num_sets, R, S, delta)
det = 2*R/sqrt(set_size);
nondet = sqrt(log(2*(2*num_sets + 1) / delta) / (2 * set_size));
bnd = ((num_sets + sqrt(num_sets))*ortho*(det + nondet)) / S;
end