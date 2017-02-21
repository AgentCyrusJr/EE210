function t = feedforward( h_raw, CNN )
%%%%%%%%%%%%%%%%%
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
h = partition_function( h_raw, 1, CNN);
for c = 1 : CNN.C
    for d = 1 : CNN.D(c)
        z = sum(h'*CNN.W{d,c},2)- sum(CNN.theta{d,c})*ones(CNN.P(c),1);
        y = tanh(z);
        t = pooling_function( y ,c, CNN );
        h = partition_function( t, c+1, CNN);
    end
end


end

