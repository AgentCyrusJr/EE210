input_size = 64;
L = 2;
D = [1,1];
P = [8,2];
Pp = [2,2];
CNN = initial_CNN(input_size, L,  D, P, Pp);
h_raw = [1:64]';
t= feedforward( h_raw, CNN );