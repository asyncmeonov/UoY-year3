%N = [1,2,3,4,5,10,15,20,25];
N = [1:2:30];
% N = [10];
%transpose for matlab
Xm = X{:,:}.';
Ym = ohY.';
train_results = zeros(1, length(N)); % vector of pairs for neurons + result

for n = 1:length(N)
    net = patternnet(N(n));
    [net, tr] = train(net, Xm, Ym);
    
    Xtest = Xm(:,tr.testInd);
    Ytest = Ym(:,tr.testInd);
    
    predY = net(Xtest);
    %evaluating against testperf because it is the only unbiased dataset
    %todo, use the same extraction and get the confusion matrix values for
    %that to compare alongside the Cross entropy
    
    perf_train = perform(net, Ytest, predY);
    train_results(n) = perf_train;
%     plotconfusion(Ytest, predY);
end
res = train_results.'








% Conver to one-hot encoding for patternnet
% ohY = zeros(length(Y.Variables),10);
% for n = 1:length(Y.Variables)
%     ohY(n,Y.CLASS(n)) = 1;
% end