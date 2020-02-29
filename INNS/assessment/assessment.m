% N = [1,2,3,4,5,10,15,20,25];
N = [1:1:100];
% N = [10];
%transpose for matlab
Xm = X{:,:}.';
% Xm = X_SisPorto.';
Ym = ohY.';
test_CE = zeros(1, length(N)); % vector of all the cross-entropy values against the test set
test_conf = zeros(1, length(N)); % vector of all the fractions of samples missclasified in the test set
best_model = struct('neur', NaN, 'CE', Inf, 'net', NaN, 'Y', NaN, 'predY', NaN);
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
    
    if test_CE(n) < best_model.CE
        best_model.CE = test_CE(n);
        best_model.net = net;
        best_model.Y = Ytest;
        best_model.predY = predY;
        best_model.neur = n;
    end 
end

saveFigs('all','fig', N, test_CE, test_conf, best_model);

function saveFigs(nameX, filetype, N, test_CE, test_conf, best_model)
  path = strcat('figures\',num2str(length(N)),'n_',nameX,'X');
  mkdir(path);
  figure('Name', 'Cross Entropy loss accross neurons');
  plot(N, test_CE);
  saveas(gcf, fullfile(path, 'delta_ce'), filetype);

  figure('Name', '% misclassified across neurons');
  plot(N, test_conf);
  saveas(gcf,fullfile(path, 'delta_misclass_rate'), filetype);

  figure('Name', 'Best model Error Histogram')
  ploterrhist(best_model.Y - best_model.predY);
  saveas(gcf,fullfile(path, 'best_err_hist'), filetype);

  figure('Name', 'Best model Conf Matrix')
  plotconfusion(best_model.Y, best_model.predY);
  saveas(gcf,fullfile(path, 'best_conf_matrix'), filetype);
  
  save(fullfile(path,'best_model'), 'best_model');
end





