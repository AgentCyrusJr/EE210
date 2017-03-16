function y = activation_nn_fun(z,l)
    if(l~=3)
        y = 1.7159*tanh(2*z/3);
     else
        y = sigmfb(z);
     end
end