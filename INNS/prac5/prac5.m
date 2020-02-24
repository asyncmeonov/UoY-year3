X = Xfit;
Y = Yfit;

N = [1,2,3,4,5,10,15,20,25];

train_results = []; % vector of pairs for neurons + result
test_results = [];

%if we reduce the width below 1 it becomes jagged
for n = N
    net = newrb(X, Y, 0.0, 1, n, 1);
    pred_y = net(X); %this is the same as sim(net,X)
    perf_train = perform(net, Y, pred_y);
    train_results = [train_results perf_train];
    
   % plot_approx(X, pred_y, n);
    pred_y2 = net(X2fit);
    perf_test = perform(net, Y2fit, pred_y2);
    test_results = [test_results perf_test];
    %plot_approx(X2fit, pred_y2, n);
end


s_results = [];

spread = (0.01:0.5:10);
for s = spread
  spread_net = newrb(X,Y,0.0,s,20,1);
  pred_y_s = spread_net(X);
  perf_s = perform(spread_net, Y, pred_y_s);
  s_results = [s_results perf_s];
end

function plot_approx(X, predY, n)
%real function
figure, hold on
  x = (-10:0.01:10);
  y = sin(x).*cos(x/5);
  plot(x,y);
%our approximation 
  plot(X,predY);
  title([num2str(n) ' neuron RBF']);
  hold off
end

