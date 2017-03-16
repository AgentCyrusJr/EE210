
function [y1,Z_c,Y_c,Z_nn,Y_nn,H] = feedforward( H_raw, CNN )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Feedforward: construct a fully-connected feedforward network
% Parameters: raw matrix---H_raw, CNN structure---CNN
% Return: output of nodes
% Details: /
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
H = cell(max(CNN.D),CNN.C+1);
[~,depth] = size(H_raw);
H_raw0 = [];
for i =1:depth
    H_raw0 = [H_raw0, add_padding_margin(H_raw(:,i),CNN.padding_margin(1))];
end

for d = 1:CNN.D(1)
H{d,1} = CNN.partition( H_raw0(:,d), 1, CNN,[]); % Hn{d, 1} denotes Hn{d, 0} 
end
Z_c = cell(1, CNN.C);
Y_c = cell(1, CNN.C);
td = [];
for c = 1 : CNN.C
    td=[];
    Z_temp=[];
    Y_temp=[];
    %for d = 1 : CNN.D(c)
    for d = 1 : CNN.D(c+1)  
        z = 0;
        for j  = 1: CNN.D(c) 
             z = z + H{j,c}'*CNN.W{d,c}(:,j)- CNN.theta{d,c}(j)*ones(CNN.P(c),1);
        end
        Z_temp = [Z_temp , z];
        y = CNN.activation_corr(z,c);
        Y_temp = [Y_temp , y];
        t = CNN.pooling( y ,c, CNN, 'max');
        td = [td,t];
        
        % 02/26/2017 @Yahya : Edited this, since we don't need the last partitioning.
        if c < CNN.C
            H{d,c+1} = CNN.partition( add_padding_margin(t,CNN.padding_margin(c+1)), c+1, CNN,[]);
        end
    end
    Z_c{c} = Z_temp;
    Y_c{c} = Y_temp;
    % @Yahya: Renamed these variables to explain they are from corr.
end
[l,w] = size(td);
y1= reshape(td,l*w,1);

Y_nn = cell(1,CNN.L);
Z_nn = cell(1,CNN.L);
%@Shashank 2/28/2017 Adding statement for value of Y_nn{1} which is
%required for grad descent
Y_nn{1}=y1;
for l = 1 : CNN.L-1
    Z_nn{l+1} = CNN.Wnn{l}*y1 - CNN.thetann{l};
    y1 = CNN.activation_nn(Z_nn{l+1},l+1);
    Y_nn{l+1} = y1;

end

end

% function [Z,Y,y1] = feedforward( H_raw, CNN )
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Feedforward: construct a fully-connected feedforward network
% % Parameters: raw matrix---H_raw, CNN structure---CNN
% % Return: output of nodes
% % Details: /
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% H = cell(max(CNN.D),CNN.C+1);
% for d = 1:CNN.D(1)
% H{d,1} = CNN.partition( H_raw(:,d), 1, CNN); % Hn{d, 1} denotes Hn{d, 0} 
% end
% Z = cell(1, CNN.C);
% Y = cell(1, CNN.C);
% td = [];
% for c = 1 : CNN.C
%     td=[];
%     Z_temp=[];
%     Y_temp=[];
%     for d = 1 : CNN.D(c+1)
%         z = 0;
%         for j  = 1: CNN.D(c)
%             
%             ttt = CNN.theta{d,c}(j)*ones(CNN.P(c),1);
%             size(H{j,c}')
%             size(CNN.W{d,c}(:,j))
%             hhh = H{j,c}'*CNN.W{d,c}(:,j);
%             z = z + hhh- ttt;
%         end
%         Z_temp = [Z_temp , z];
%         y = CNN.activation_corr(z,c);
%         Y_temp = [Y_temp , y];
%         t = CNN.pooling( y ,c, CNN, 'max');
%         td = [td,t];
%         H{d,c+1} = CNN.partition( t, c+1, CNN);
%     end
%     Z{c} = Z_temp;
%     Y{c} = Y_temp;
% end
% [l,w] = size(td);
% y1= reshape(td,l*w,1);
% for l = 1 : CNN.L-1
%     z1 = CNN.Wnn{l}*y1 - CNN.thetann{l};
%     y1 = CNN.activation_nn(z1,l);
% end
% 
% end

