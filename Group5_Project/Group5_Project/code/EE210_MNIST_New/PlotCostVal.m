load costVal.mat
subplot(3,1,1)
plot(1:length(costVal1),costVal1);
xlabel('epoch');
ylabel('cost function value');
h_legend = legend('Batchsize = 1');
set(h_legend,'FontSize',14);
subplot(3,1,2)
plot(1:length(costVal5),costVal5);
xlabel('epoch');
ylabel('cost function value');
h_legend = legend('Batchsize = 5');
set(h_legend,'FontSize',14);
subplot(3,1,3)
plot(1:length(costVal10),costVal10);
xlabel('epoch');
ylabel('cost function value');
h_legend = legend('Batchsize = 10');
set(h_legend,'FontSize',14);
suptitle('cost function value - iteration for different batch size')