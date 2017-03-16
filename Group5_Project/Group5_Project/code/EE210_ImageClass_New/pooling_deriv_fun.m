function [pool_deriv_out] = pooling_deriv_fun(CNN,y,c,pooling_method)
% The input to the function is already permuted column vector. 
% The function just divides the input per pooling size and put the
% derivative for each.
% The output has the same size as the input.

y_reshaped = reshape(y,CNN.pooling_size(c),length(y)/CNN.pooling_size(c));


if ( strcmp(pooling_method, 'mean') )
    pool_deriv_out = ones(size(y_reshaped)) / CNN.pooling_size(c);
elseif ( strcmp(pooling_method, 'max') )
    y_reshaped_max = repmat(max(y_reshaped),size(y_reshaped,1),1);
    % Generates a matrix where every element in a column is replaced with
    % the max of that column.
    pool_deriv_out = y_reshaped >= y_reshaped_max;
    pool_deriv_out = pool_deriv_out ./ repmat(sum(pool_deriv_out),size(pool_deriv_out,1),1);
end


pool_deriv_out = pool_deriv_out(:);




end