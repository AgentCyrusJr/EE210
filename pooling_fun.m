function t = pooling_fun( y ,c, CNN, pooling_method)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Pooling function: reduce each feature map to a smaller vector
% Parameters: feature map---y, index of convolution layer---c, CNN
% structure---CNN, pooling strategy---pooling_method
% Return: a smaller vector representation of Pp*1---t
% Details: depending on c, a different pooling can be applied if needed.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% permutation
y0 = CNN.permutation_matrix{c}*y;

% pooling
helper_matrix = reshape(y0, length(y0)/CNN.Pp(c), CNN.Pp(c));

% We provide different pooling strategies, currently we have
% 1. mean-pooling 
% 2. max-pooling

if ( strcmp(pooling_method, 'mean') )
    t = mean(helper_matrix, 1)';
else if ( strcmp(pooling_method, 'max') )
    t = max(helper_matrix)';
    end
end
end

