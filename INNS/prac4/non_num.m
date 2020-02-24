fruit = fruit{:,:};
net = feedforwardnet(2);
net = train(net, fruit(:,2), fruit(:,1));