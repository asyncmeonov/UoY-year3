N = [1,2,3,4,5,10,15,20,25];
L = N;
% N = [1:1:100];
%N = [10];
%transpose for matlab
Xm = X{:,:}.';
% Xm = X_SisPorto.';
%Xm = X{:,:}(:,[20,2,21,5]).'; % first 4 important features
Ym = ohY.';
test_CE = zeros(1, length(N)); % vector of all the cross-entropy values against the test set
test_conf = zeros(1, length(N)); % vector of all the fractions of samples missclasified in the test set
best_model = struct('neur', NaN, 'CE', Inf, 'net', NaN, 'Y', NaN, 'predY', NaN);

%single hidden layer
% for n = 1:length(N)
%     net = patternnet(N(n));
%     [net, tr] = train(net, Xm, Ym);
%     
%     %extract only the entries used for test
%     Xtest = Xm(:,tr.testInd);
%     Ytest = Ym(:,tr.testInd);
%     
%     predY = net(Xtest);
%     %evaluating against testperf because it is the only unbiased dataset
%     %todo, use the same extraction and get the confusion matrix values for
%     %that to compare alongside the Cross entropy
%     test_CE(n) = perform(net, Ytest, predY);
%     test_conf(n) = confusion(Ytest, predY);
%     
%     if test_CE(n) < best_model.CE
%         best_model.CE = test_CE(n);
%         best_model.net = net;
%         best_model.Y = Ytest;
%         best_model.predY = predY;
%         best_model.neur = n;
%     end 
% end

%multiple hidden layers
for n = 1:length(N)
    idL = L < N(n); % subsequent hl must always have less neurons than the preceding
    HL = L(idL);
    for l = 1:length(HL)
        
        net = patternnet([N(n),HL(l)]);
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
end

saveFigs('two_layer','fig', N, test_CE, test_conf, best_model, true);

function saveFigs(nameX, filetype, N, test_CE, test_conf, best_model, isTest)
  if isTest
    path = strcat('figures\testfigs\',num2str(length(N)),'n_',nameX,'X');
  else
    path = strcat('figures\',num2str(length(N)),'n_',nameX,'X');
  end
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
  
  jframe = view(best_model.net);
  saveJframe(jframe, fullfile(path, 'best_architecture'));
  
  save(fullfile(path,'best_model'), 'best_model');
end

function saveJframe(frame, filepath)
  %# create it in a MATLAB figure
  hFig = figure('Menubar','none', 'Position',[100 100 565 166]);
  jpanel = get(frame,'ContentPane');
  [~,h] = javacomponent(jpanel);
  set(h, 'units','normalized', 'position',[0 0 1 1])

%# close java window
  frame.setVisible(false);
  frame.dispose();

%# print to file
  set(hFig, 'PaperPositionMode', 'auto');
  saveas(hFig, strcat(filepath, '.png'));

%# close figure
  close(hFig)
end





