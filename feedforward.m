function [Z,Y,y1] = feedforward( H_raw, CNN )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Feedforward: construct a fully-connected feedforward network
% Parameters: raw matrix---H_raw, CNN structure---CNN
% Return: output of nodes
% Details: /
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

H = cell(max(CNN.D),CNN.C+1);
for d = 1:CNN.D(1)
H{d,1} = CNN.partition( H_raw(:,d), 1, CNN); % Hn{d, 1} denotes Hn{d, 0} 
end
Z = cell(1, CNN.C);
Y = cell(1, CNN.C);
td = [];
for c = 1 : CNN.C
    td=[];
    Z_temp=[];
    Y_temp=[];
    for d = 1 : CNN.D(c)
        z = 0;
        for j  = 1: CNN.D(c)
        z = z + H{j,c}'*CNN.W{d,c}(:,j)- CNN.theta{d,c}(j)*ones(CNN.P(c),1);
        end
        Z_temp = [Z_temp , z];
        y = CNN.activation_corr(z,c);
        Y_temp = [Y_temp , y];
        t = CNN.pooling( y ,c, CNN, 'max');
        td = [td,t];
        H{d,c+1} = CNN.partition( t, c+1, CNN);
    end
    Z{c} = Z_temp;
    Y{c} = Y_temp;
end
[l,w] = size(td);
y1= reshape(td,l*w,1);
for l = 1 : CNN.L-1
    z1 = CNN.Wnn{l}*y1 - CNN.thetann{l};
    y1 = CNN.activation_nn(z1,l);
end

end

