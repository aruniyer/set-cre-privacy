function [val] = beta_prime_test(eps, m, n, x)
val = log(m-1) - betaln(m, n) + (m-2).*log(x) - (m+n).*log(1+x) + safe_log(1/n - exp(eps).*x./(m-1));
end

function [out] = safe_log(v)
out = zeros(size(v));
out(v <= 0) = -Inf;
out(v > 0) = log(v(v > 0));
end