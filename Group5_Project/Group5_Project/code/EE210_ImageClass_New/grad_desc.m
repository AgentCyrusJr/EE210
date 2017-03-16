function [CNN_update] = grad_desc(CNN,H_raw,gamma,mu,rho1,rho2,Batch_size,CNN_update)

if ~exist('CNN_update','var')
    CNN_update = CNN;
end

if ~exist('Batch_size','var')
    Batch_size = 1;
end


D=CNN.D;
[~,~,~,~,Y_nn,H] = feedforward( H_raw, CNN );

[d_nn,Delta_c,~,~] = CalculateDerivatives(CNN,H_raw,gamma);

%A=cell(CNN.D(CNN.C+1),1);
W=cell(CNN.C,1);
W_u=cell(CNN.C,1);
theta=cell(CNN.C,1);

for c = 1 : CNN.C
    temp=reshape(CNN.W{1,c},[],1);
     A=temp;
     
    temp_u=reshape(CNN_update.W{1,c},[],1);
     A_u=temp_u;
    for d = 2 : CNN.D(c+1)
        temp=reshape(CNN.W{d,c},[],1);
        A=[A temp];
        
        temp_u=reshape(CNN_update.W{d,c},[],1);
        A_u=[A_u temp_u];
    end
    W{c}=A;
    W_u{c}=A_u;
end

for c = 1 : CNN.C
    temp=reshape(CNN.theta{1,c},[],1);
     B=temp;
     
     temp_u=reshape(CNN_update.theta{1,c},[],1);
     B_u=temp_u;
    for d = 2 : CNN.D(c+1)
        temp=reshape(CNN.theta{d,c},[],1);
        B=[B temp];
        
        temp_u=reshape(CNN_update.theta{d,c},[],1);
        B_u=[B_u temp_u];
    end
    theta{c}=B;
    theta_u{c}=B_u;
end

for l=1:CNN.L-1
    CNN_update.Wnn{l}=CNN_update.Wnn{l}-(2*mu*rho2/Batch_size)*CNN.Wnn{l}-mu*d_nn{l+1}*Y_nn{l}'/Batch_size;
    CNN_update.thetann{l}=CNN_update.thetann{l}+mu*d_nn{l+1}/Batch_size;
end


for c=1:CNN.C
    a=ones(D(c+1),1);
    
    H_concat = [];
    for d = 1:D(c)
        H_concat = [H_concat;H{d,c}];
    end
    
    b=ones(D(c),1);
    d=ones(CNN.P(c),1);
    e=b*d';
    %for d1=1:CNN.D(c)
     %   r=isempty(H(d1,c));
     %   if r==0
     %   H_app{d1}=H{d1,c};
     %   end
    %end
    k=kron(a',H_concat);
    
    k1=kron(a',e);
    W_u{c}=W_u{c}-(2*mu*rho1/Batch_size)*W{c}-mu*k*Delta_c{c}/Batch_size;
    theta_u{c}=theta_u{c}+mu*k1*Delta_c{c}/Batch_size;
end


% Wnn=CNN.Wnn;
%W=W;
%theta=CNN.theta;
% thetann=CNN.thetann;

theta_fin=theta;

W_fin=cell(max(CNN.D),CNN.C);
%theta_fin=cell(max(CNN.D),CNN.C);



% padded_layer_input_size = (sqrt(CNN.layer_input_size)+2.*CNN.padding_margin).^2;

for c = 1:CNN.C
    
for d = 1 : CNN.D(c+1)  %CNN.D(c+1)
    W_fin{d,c}= reshape(W_u{c}(:,d),[], CNN.D(c));
    theta_fin{d,c}=reshape(theta_u{c}(:,d),CNN.D(c),1);
end 
end

 CNN_update.W = W_fin;

 CNN_update.theta = theta_fin;

end


