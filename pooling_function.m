function t = pooling_function( y ,c, CNN )
% The pooling function, takes the vector y and the index of the correlation layer c.
% Depending on c, a different pooling can be applied if needed.
%UNTITLED10 Summary of this function goes here
%   Detailed explanation goes here

% permutation
y0 = CNN.permutation_matrix{c}*y;

% pooling
helper_matrix = reshape(y0, length(y0)/CNN.Pp(c), CNN.Pp(c));
t = mean(helper_matrix, 1)';



end

