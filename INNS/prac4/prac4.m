net = patternnet(10);

trans_data = uc_data;

for row= 1:length(uc_data)
    if( trans_data(row,3) == -1)
        trans_data(row,3) = 0;
    end  
end

trans_data = trans_data.'; %transpose

input = trans_data([1,2],:);
output = trans_data(3,:);

iter = 25;
perf = [];

for n = 1:iter
    net = init(net);
    [net, tr] = train(net, input, output);
    %perf = [perf tr.best_vperf];
    perf(n) = perform(net, input, output);
end

avg_perf = mean(perf)


perform(net, uc_test(:,[1,2]), uc_test(:,3))