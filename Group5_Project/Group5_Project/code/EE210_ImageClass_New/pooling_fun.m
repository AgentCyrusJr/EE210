function [t,permutation_matrix,permutation_fix] = pooling_fun( y ,c, CNN, pooling_method)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Pooling function: reduce each feature map to a smaller vector
% Parameters: feature map---y, index of convolution layer---c, CNN
% structure---CNN, pooling strategy---pooling_method
% Return: a smaller vector representation of Pp*1---t
% Details: depending on c, a different pooling can be applied if needed.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% permutation



% [permutation_matrix,permutation_fix] = permutation_matrix_image([2 2],[2 2],[25 25]);


if ( strcmp(pooling_method, 'init') )
    [permutation_matrix,permutation_fix] = permutation_matrix_image(CNN,c);
    t = [];
    
elseif ( strcmp(pooling_method, 'inv_permutation') )
    
    t = [CNN.permutation_matrix{c};CNN.permutation_fix{c}].'*y;

else
    
    % permutation_matrix = eye(CNN.P(c));
    y0 = CNN.permutation_matrix{c}*y;
    
    % pooling
    helper_matrix = reshape(y0, length(y0)/CNN.Pp(c), CNN.Pp(c));
    
    % We provide different pooling strategies, currently we have
    % 1. mean-pooling
    % 2. max-pooling
    
    if ( strcmp(pooling_method, 'mean') )
        t = mean(helper_matrix, 1)';
    elseif ( strcmp(pooling_method, 'max') )
        t = max(helper_matrix,[],1)';
    elseif ( strcmp(pooling_method, 'permutation') )
        t = y0;
        
    end

end

end


function [permutation_matrix,permutation_fix] = permutation_matrix_image(CNN,c)
image_size = sqrt(CNN.P(c))*ones(1,2);
filtersize = CNN.pool_filtersize(c)*ones(1,2);
stride = filtersize;

    permutation_matrix = [];
    stored_selected_indices = [];
    count_pools = 0;
    for w = 1:stride(2):image_size(2)-filtersize(2)+1
        for h =1:stride(1):image_size(1)-filtersize(1)+1
            count_pools = count_pools+1;
            indexes_selected = repmat((h:h+filtersize(1)-1).',1,filtersize(2));
            col_fixes = image_size(1)*((0:filtersize(2)-1) + w-1); 
            indexes_selected = indexes_selected + repmat(col_fixes,filtersize(1),1);
            num_indexes_selected = numel(indexes_selected);
%             subPerm = sparse(1:num_indexes_selected,indexes_selected(:),ones(1,num_indexes_selected),num_indexes_selected,prod(image_size));
            stored_selected_indices = [stored_selected_indices;indexes_selected(:)];
            %subPerm = zeros(prod(filter_size),prod(image_size));
            %subPerm(:,indexes_selected) = eye(prod(filter_size));
%             permutation_matrix = [permutation_matrix;subPerm];           
            
        end
        len_selected = length(stored_selected_indices);
        permutation_matrix = sparse(1:len_selected,stored_selected_indices,ones(1,len_selected),len_selected,prod(image_size));
        
    end
    %%%% Fix for unchosen elements when doing inv_perm
    unselected_indices = setdiff(1:CNN.P(c),stored_selected_indices);
    permutation_fix = sparse(1:numel(unselected_indices),unselected_indices,ones(1,numel(unselected_indices)),numel(unselected_indices),prod(image_size));
%     S = sum(permutation_matrix);
%     permutation_fix = zeros(length(S(S==0)),prod(image_size));
%     permutation_fix(:,S==0) = eye(length(S(S==0)));
    
       
end