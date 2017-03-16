clear
clc 

%% Data Preprocessing
test_data = load('test.mat');
train_data = load('train.mat');

Indices = crossvalind('Kfold', 60000, 24);
trainset = train_data.train;
testset = test_data.test;

image_size = 32;
images_train_all=zeros(image_size^2,60000);
labels_train_all = trainset.labels;

images_test_all=zeros(image_size^2,10000);
labels_test_all = testset.labels;

for i = 1:60000
    temp =  trainset.images(:,:,i);
    temp = [zeros(28,2),temp,zeros(28,2)];
    temp = [zeros(2,image_size);temp;zeros(2,image_size)];
    images_train_all(:,i) = double(reshape(temp,image_size^2,1))./255;
end

for i = 1:10000
    temp =  testset.images(:,:,i);
    temp = [zeros(28,2),temp,zeros(28,2)];
    temp = [zeros(2,image_size);temp;zeros(2,image_size)];
    images_test_all(:,i) = double(reshape(temp,image_size^2,1))./255;
end

images_train = images_train_all(:,find(Indices>=2&Indices<=5));
labels_train = labels_train_all(find(Indices>=2&Indices<=5));

images_valid = images_train_all(:,find(Indices==1));
labels_valid = labels_train_all(find(Indices==1));

images_test = images_test_all;
labels_test = labels_test_all;
%%
%% Determine HyperParameters

% size of filter in each layer
filter_size = [3,3];

% pooling filter size
pool_filtersize = [2,2];

% # of neuron nodes
N_neuron = [200,10];

% value of sride in each layer
stride = [2,1]; 

% value of padding margin in each layerH
%padding_margin = [1,1];
padding_margin = (filter_size - 1) / 2;

% # of depth in each layer
D = [1,5,5];

% step size
mu = 0.01;

% regularization coefficient for Wc
rho1 = 0.01;

% regularization coefficient for Wl
rho2 = 0.01;

% batch size
Batch_size = 10;

% maximum number of epoch
MaxEpoch = 100;

%%%% Automatically calculate the size of P and Pp
P = zeros(1,length(D)-1);
Pp = zeros(1,length(D)-1);
P(1) = floor((image_size + 2*padding_margin(1) - filter_size(1)) / stride(1) + 1 ).^2;
Pp(1) = floor( sqrt(P(1)) / pool_filtersize(1) ).^2;

for i=2:length(P)
    P(i) =  floor((sqrt(Pp(i-1)) + 2*padding_margin(i) - filter_size(i)) / stride(i) + 1 ).^2;
    Pp(i) = floor( sqrt(P(i)) / pool_filtersize(i) ).^2;
end
%%
%% Training 

% Initialization of the structure of CNN
CNN = initial_CNN( D , P , Pp , N_neuron , filter_size , padding_margin , stride, pool_filtersize,image_size*ones(1,2));
[~,N_train] = size(images_train);
costVal = [];
train_accuracy = [];

CNN_orig = CNN;

images_train_orig = images_train;
labels_train_orig = labels_train;

CNN_update = CNN;
tic
for epoch = 1:MaxEpoch
%     r_perm = randperm(N_train);
%     images_train = images_train_orig(:,r_perm);
%     labels_train = labels_train_orig(r_perm);
      r_sample = datasample(1:N_train,Batch_size);
      images_train = images_train_orig(:,r_sample);
      labels_train = labels_train_orig(r_sample);
      
    for index=1:Batch_size;
        

        gamma = zeros(10,1);
        gamma(labels_train(index)+1)=1;
        H_raw = images_train(:,index);
        [y1,~,~,~,~,~]= feedforward( H_raw, CNN );
        

        %%%%%%%% If you want to run Stochastic Gradient descent, set Batch_size = 1
        CNN_update = grad_desc(CNN,H_raw,gamma,mu,rho1,rho2,Batch_size,CNN_update);
        if index == Batch_size
            CNN = CNN_update;
        end
        
      

    end
    cost = - sum(gamma .* log(y1)+(1-gamma).* log(1-y1));
    costVal = [costVal , cost];
       [train_accuracy1 , ~] = testCNN(images_train_orig,labels_train_orig,CNN);
       train_accuracy = [train_accuracy , train_accuracy1];
    epoch
end
toc

%% Testing/Validation
[valid_accuracy , store_decisions] = testCNN(images_valid,labels_valid,CNN);
%[test_accuracy , ~] = testCNN(images_test,labels_test,CNN);
%%

%% Plot
figure;
plot(1:length(costVal),costVal);
xlabel('epoch');
ylabel('Cost Function Value');
title('Cost Function Value - epoch');
figure;
plot(1:100:100*length(train_accuracy),train_accuracy);
xlabel('epoch');
ylabel('Training Accuracy');
title('Training Accuracy - epoch');
%%


