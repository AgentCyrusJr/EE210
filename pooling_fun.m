function t = pooling_fun( y ,c, CNN )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Pooling function: reduce each feature map to a smaller vector
% Parameters: feature map---y, index of convolution layer---c, CNN struture---CNN
% Return: a smaller vector representation of Pp*1---t
% Details: depending on c, a different pooling can be applied if needed.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% permutation
y0 = CNN.permutation_matrix{c}*y;

% pooling
helper_matrix = reshape(y0, length(y0)/CNN.Pp(c), CNN.Pp(c));
t = mean(helper_matrix, 1)';

end

