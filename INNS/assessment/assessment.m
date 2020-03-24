N = repelem([37],500);
% N = [1:100];

feature_subset_name = 'regularised_no_corr_X';

% t_algorithms = ["trainrp","trainscg","traincgb","traincgf","traincgp","trainoss","traingdx"];
t_algorithms = ["traincgb"];

% t_perf = zeros(length(N),length(t_algorithms));
% t_time = zeros(length(N),length(t_algorithms));

%1	2	3	4	5	6	7	8       9       10      11  	12  	13  14  15  	16  	17      18      19      20          21
%LB	AC	FM	UC	DL	DS	DP	ASTV	MSTV	ALTV	MLTV	Width	Min	Max	Nmax	Nzeros	Mode	Mean	Median	Variance	Tendency

Xm = Xo_NSP(:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,19,20,21]).'; %regularised reduced correlated

Ym = preprocess.one_hot_encode(Yo_NSP).';
test_CE = zeros(1, length(N)); % vector of all the cross-entropy values against the test set
test_conf = zeros(1, length(N)); % vector of all the fractions of samples missclasified in the test set
time = zeros(length(N),1); % vector of how long each network took
best_model = struct('neur', NaN, 'CE', Inf, 'net', NaN, 'Y', NaN, 'predY', NaN, 'tr', NaN);

%single hidden layer
for t = 1:length(t_algorithms)
    for n = 1:length(N)
        [net, tr , Xtest, Ytest, predY] = train_single_layer(N(n), Xm, Ym, t_algorithms(t));

        %evaluating against testperf because it is the only unbiased dataset
        test_CE(n) = perform(net, Ytest, predY);
        test_conf(n) = confusion(Ytest, predY);
        time(n) = sum(tr.time);

        if test_CE(n) < best_model.CE
            best_model.CE = test_CE(n);
            best_model.net = net;
            best_model.Y = Ytest;
            best_model.predY = predY;
            best_model.neur = N(n);
            best_model.tr = tr;
        end
    end
    
%      name = strcat(num2str(now),'_',num2str(length(N)),'n_', net.divideFcn,'_', net.trainFcn,'_',net.performFcn,'_',feature_subset_name);
%      saveFigs(name,'fig', N, test_CE, test_conf, best_model, time, false);
end

function [net,tr, Xtest, Ytest, predY] = train_single_layer(n, Xm, Ym, trainAlgo)
    net = patternnet(n);
    net.trainFcn = trainAlgo;
    net.performFcn = 'crossentropy';
    net.inputs{1}.processFcns{2} = 'mapstd'; % 0 mean, unit variance standartisation. For [-1 1] use mapminmax
    net.divideParam.trainRatio = 0.7;  %default
    net.divideParam.valRatio = 0.15;   %default
    net.divideParam.testRatio = 0.15;  %default
    
    [net, tr] = train(net, Xm, Ym);
    
    %extract only the entries used for test
    Xtest = Xm(:,tr.testInd);
    Ytest = Ym(:,tr.testInd);
    
    predY = net(Xtest);  
end

function saveFigs(nameX, filetype, N, test_CE, test_conf, best_model, time, isTest)
  if isTest
    path = strcat('figures\testfigs\',nameX);
  else
    path = strcat('figures\',nameX);
  end
  mkdir(path);
  figure('Name', 'Cross Entropy loss accross neurons');
  plot(N, test_CE);
  saveas(gcf, fullfile(path, 'delta_ce'), filetype);

  figure('Name', '% misclassified across neurons');
  plot(N, test_conf);
  saveas(gcf,fullfile(path, 'delta_misclass_rate'), filetype);
  
  figure('Name', 'Training time')
  plot(N, time);
  saveas(gcf,fullfile(path, 'delta_time'), filetype);

  figure('Name', 'Best model Error Histogram')
  ploterrhist(best_model.Y - best_model.predY);
  saveas(gcf,fullfile(path, 'best_err_hist'), filetype);

  figure('Name', 'Best model Conf Matrix')
  plotconfusion(best_model.Y, best_model.predY);
  saveas(gcf,fullfile(path, 'best_conf_matrix'), filetype);
  
  jframe = view(best_model.net);
  saveJframe(jframe, fullfile(path, 'best_architecture'));
  
  save(fullfile(path,'best_model'), 'best_model');
  close all
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



% %multiple hidden layers
% for n = 1:length(N)
%     idL = L < N(n); % subsequent hl must always have less neurons than the preceding
%     HL = L(idL);
%     for l = 1:length(HL)
%         
%         net = patternnet([N(n),HL(l)]);
%         [net, tr] = train(net, Xm, Ym);
% 
%         %extract only the entries used for test
%         Xtest = Xm(:,tr.testInd);
%         Ytest = Ym(:,tr.testInd);
% 
%         predY = net(Xtest);
%         %evaluating against testperf because it is the only unbiased dataset
%         %todo, use the same extraction and get the confusion matrix values for
%         %that to compare alongside the Cross entropy
%         test_CE(n) = perform(net, Ytest, predY);
%         test_conf(n) = confusion(Ytest, predY);
% 
%         if test_CE(n) < best_model.CE
%             best_model.CE = test_CE(n);
%             best_model.net = net;
%             best_model.Y = Ytest;
%             best_model.predY = predY;
%             best_model.neur = n;
%         end 
%     end
% end




