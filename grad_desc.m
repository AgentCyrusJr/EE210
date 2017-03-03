function [Wnn,W_fin,theta_fin,thetann] = grad_desc(CNN,H_raw,gamma)

rho1=0.5;
rho2=0.5;
mu=0.2;

D=CNN.D;
[~,~,~,~,Y_nn,H] = feedforward( H_raw, CNN );

[d_nn,Delta_c,~,~] = CalculateDerivatives(CNN,H_raw,gamma);


%A=cell(CNN.D(CNN.C+1),1);
W=cell(CNN.C,1);
theta=cell(CNN.C,1);

for c = 1 : CNN.C
    temp=reshape(CNN.W{1,c},[],1);
     A=temp;
    for d = 2 : CNN.D(c+1)
        temp=reshape(CNN.W{d,c},[],1);
        A=[A temp];
    end
    W{c}=A;
end

for c = 1 : CNN.C
    temp=reshape(CNN.theta{1,c},[],1);
     B=temp;
    for d = 2 : CNN.D(c+1)
        temp=reshape(CNN.theta{d,c},[],1);
        B=[B temp];
    end
    theta{c}=B;
end
                     

for l=1:CNN.L-1
    CNN.Wnn{l}=(1-2*mu*rho2)*CNN.Wnn{l}-mu*d_nn{l+1}*Y_nn{l}';
    CNN.thetann{l}=CNN.thetann{l}-mu*d_nn{l+1};
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
    W{c}=(1-2*mu*rho1)*W{c}-mu*k*Delta_c{c};
    theta{c}=theta{c}+mu*k1*Delta_c{c};
end


Wnn=CNN.Wnn;
%W=W;
%theta=CNN.theta;
thetann=CNN.thetann;

theta_fin=theta;

W_fin=cell(max(CNN.D),CNN.C);
%theta_fin=cell(max(CNN.D),CNN.C);


for c = 1:CNN.C
    
for d = 1 : CNN.D(c+1)  %CNN.D(c+1)
    if c==1
    temp=reshape(W{c}(:,d),CNN.input_size/CNN.P(c), CNN.D(c));
    else
        temp=reshape(W{c}(:,d),CNN.Pp(c-1)/CNN.P(c),CNN.D(c));
    end
    W_fin{d,c}=temp;
    temp1=reshape(theta{c}(:,d),CNN.D(c),1);
    theta_fin{d,c}=temp1;
end 
end

end


