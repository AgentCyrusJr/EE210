function [d_nn,Delta_c,Y_nn,H] = CalculateDerivatives(CNN,H_raw,gamma)

% The output of this function is the 
L = CNN.L;
C = CNN.C;

D = CNN.D(2:end);

[~,Z_c,Y_c,Z_nn,Y_nn,H] = feedforward( H_raw, CNN );

%%%%% Derivatives of the Neural Network Part %%%%%%
% Yahya : This is the simpler part.

d_nn = cell(1,L);
% Create a cell that contains the different delta_l

%d_nn{L} = 2*(Y_nn{L} - gamma) .* CNN.activation_nn_deriv(Z_nn{L},L);

d_nn{L} = (Y_nn{L} - gamma);

%d_nn{2} = (CNN.Wnn{2})' * d_nn{3} .* sigmfb(Y_nn{2}).*(1-sigmfb(Y_nn{2}));
for l = L-1:-1:2
    d_nn{l} = (CNN.Wnn{l})' * d_nn{l+1} .*  CNN.activation_nn_deriv(Z_nn{l},l);
    % The recursion for computing the partial derivatives in the full network.
end





%%%%% Derivatives of the Correlation Network Part %%%%%%

d_corr = cell(1,C);
    

% v = repmat((CNN.Wnn{1}.' * d_nn{2}).',CNN.pooling_size(C),1);
% 
% for d = 1:CNN.D(C)
%     d_corr{C}(:,d) = CNN.corr_activation_deriv(Z_c{C,d},C) .* CNN.inv_permutation( v(:) .* CNN.pooling_deriv(CNN.pooling(y_c{C,:},C),C) , C);
% end
    
d_corr{C} = Z_c{C}*0;

for d = 1:D(C)
    v = CNN.Wnn{1}(:,(d-1)*CNN.Pp(C)+(1:CNN.Pp(C)))' * d_nn{2};
    upsampled_v = upsample(CNN,v,Y_c{C}(:,d),C);  % Refer to the upsampling implemenetation at the bottom of this file.
    d_corr{C}(:,d) = CNN.activation_corr_deriv(Z_c{C}(:,d),C) .* CNN.pooling(upsampled_v,C,CNN,'inv_permutation');
end

Delta_c = cell(1,C);

V_d_dnxt = cell(max(D), max(D), CNN.C);

for c = C-1:-1:1
    S_c_nxt = length(CNN.W{1,c+1}(:,1).');
    OuterKron = kron( speye(CNN.P(c+1)) , ones(S_c_nxt,1) );  
   
    padded_input_len = (sqrt(CNN.Pp(c))+2*CNN.padding_margin(c+1))^2;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%% Yahya : Replaced this with an easier one liner
%     % Creat an empty V_c for paritioning. 
%     %VERY IMPORTANT: The num of rows of V_c are determined by parameters of c+1, 
%     %while the columns are determined by c.
%     V_c= zeros(CNN.P(c+1)*S_c_nxt, padded_input_len);  
%     
%     % Fill the V matrix with ones based on the partitioning of this layer
%     partition_map = CNN.partition((1:padded_input_len).' , c+1 , CNN);
%     partition_map = partition_map(:);
%     V_c( sub2ind(size(V_c),1:size(V_c,1),partition_map.') ) = 1;  
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    d_corr{c} = Z_c{c}*0;
    
    
    for d = 1:D(c)
        v_d = 0;
        for d_nxt = 1:D(c+1)
            %%%% Computing small deltas now            
            % z^{(d_nxt,c+1)} = \sum_{d = 1}^{D(c)} [ W_change_layer_c_d_to_d_nxt  t^{(d,c)} ] 
            W_change_layer_c_d_to_d_nxt = kron(speye(CNN.P(c+1)),CNN.W{d_nxt,c+1}(:,d).') * CNN.partition_matrix{c+1} * CNN.padding_matrix{c+1};
            v_d = v_d + W_change_layer_c_d_to_d_nxt.' * d_corr{c+1}(:,d_nxt);
            
            
        end
        upsampled_v = upsample(CNN,v_d,Y_c{c}(:,d),c);  
        d_corr{c}(:,d) = CNN.activation_corr_deriv(Z_c{c}(:,d),c) .* CNN.pooling(upsampled_v,c,CNN,'inv_permutation'); 
    Delta_c{c} = blkdiag(Delta_c{c},d_corr{c}(:,d));
        
    end    
    
end


for d = 1:D(C)
    Delta_c{C} = blkdiag(Delta_c{C},d_corr{C}(:,d));
    
end


end


%%%%% The Upsampling function (DONE)
%%% Input can be of the form v, y^(d,c) and c to give us the expression of
%%% upv(v, pooling_derivative(y^(d,c)))
function [v_up] =  upsample(CNN,v,y,c)
v = repmat(v.',CNN.pooling_size(c),1);
v = v(:);
pooled_y = CNN.pooling(y,c,CNN,'permutation');
v_up = v .* CNN.pooling_deriv(CNN,pooled_y,c,'max');

end