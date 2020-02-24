for i = 1:250 %[1,3,5,10,15,20,25]
    X = S3([1,2],:);
    Y = S3(3,:);
    trainNN(i,X,Y);
end

function trainNN(neurons, X, Y)
  ffnn = feedforwardnet(neurons);
  [ffnn, tr] = train(ffnn, X,Y);
  predY = sim(ffnn,X);
  print_best(neurons,tr);
  plt(X,Y,predY,neurons,tr.best_vperf);
end

function print_best(neurons,tr)
  format = [neurons,tr.best_vperf];
  disp(format)
end

function plt(Xfit,Yfit,predY,label,performace)
  figure('Name',strcat('num neurons: ', int2str(label)));
  scatter(Xfit(1,:), Xfit(2,:), [],Yfit); %X:: real Y
  title(performace);
  hold on
 % plot(Xfit,predY,'r');  % X::predicted Y
end