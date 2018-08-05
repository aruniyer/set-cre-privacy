function [ obj, theta_root ] = epsdelta_in_r( f, v, epsilon )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

eeps = exp(epsilon);

% f1 = (m.*((m-1).*eeps+n)) ./ ((m+n).*(m-1).*eeps);
% if (f1 > 1.1)
%     f1
%     m
%     n
%     epsilon
%     pause
% end
% f2 = (n.*((n-1).*eeps+m)) / ((m+n).*m);
% if (f2 > 1.1)
%     f2
%     m
%     n
%     epsilon
%     pause
% end

f1 = (f./(f + 1)).*(((f - (f+1).^2 .* v).*(f.*eeps+ 1) - eeps.*((f+1).^3).*v)./((f.^2 - f.*((f+1).^2).*v - ((f+1).^3).*v).*eeps));
f2 = (1./(f + 1)).*(((f - ((f+1).^2) .* v).*(f + eeps) - eeps.*((f+1).^3).*v)./((f.^2 - f.*((f+1).^2).*v)));

% theta_root = (m - 1)./(m - 1 + n.*eeps);
theta_root = 0;
obj = max(f1, f2);

end

