%train ffnn in the console cause matlab is funky
%my_mlp is the nn
% you can only train an untrained nn


%for i = [1,3,5,10,15,20,25]
%    trainNN(i,Xfit,Yfit);
%end

function trainNN(neurons, X, Y)
  ffnn = feedforwardnet(neurons);
  [ffnn, tr] = train(ffnn, X,Y);
  predY = sim(ffnn,X);
  plt(X,Y,predY,neurons,tr.best_perf);
end

function plt(Xfit,Yfit,predY,label,performace)
  figure('Name',strcat('num neurons: ', int2str(label)));
  plot(Xfit, Yfit, 'b'); %X:: real Y
  title(performace);
  hold on
  plot(Xfit,predY,'r');  % X::predicted Y
end