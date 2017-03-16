function y = sigmfb(x)
 y = 1./ (1+exp(-x));
 % sigmf(x,[1,0]);
 y(abs(y-1)<1e-16) = 1-1e-16;
 y(abs(y)<1e-16) = 1e-16;
end
