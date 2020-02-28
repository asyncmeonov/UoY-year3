% N = [1,2,3,4,5,10,15,20,25];
N = [1:2:100];
% N = [10];
%transpose for matlab
% Xm = X{:,:}.';
Xm = X_SisProto.';
Ym = ohY.';
test_CE = zeros(1, length(N)); % vector of all the cross-entropy values against the test set
test_conf = zeros(1, length(N)); % vector of all the fractions of samples missclasified in the test set

for n = 1:length(N)
    net = patternnet(N(n));
    [net, tr] = train(net, Xm, Ym);
    
    %extract only the entries used for test
    Xtest = Xm(:,tr.testInd);
    Ytest = Ym(:,tr.testInd);
    
    predY = net(Xtest);
    %evaluating against testperf because it is the only unbiased dataset
    %todo, use the same extraction and get the confusion matrix values for
    %that to compare alongside the Cross entropy
    test_CE(n) = perform(net, Ytest, predY);
    test_conf(n) = confusion(Ytest, predY);
%     plotconfusion(Ytest, predY);
end

figure('Name', 'Cross Entropy loss accross neurons');
plot(N, test_CE);

figure('Name', '% misclassified across neurons');
plot(N, test_conf);







% Conver to one-hot encoding for patternnet
% ohY = zeros(length(Y.Variables),10);
% for n = 1:length(Y.Variables)
%     ohY(n,Y.CLASS(n)) = 1;
% end