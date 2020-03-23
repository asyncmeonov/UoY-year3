% plot feature interaction
% --------------------------
% features = CTG{:,:};
% target = Y{:,:};
% for i = 1:length(features(1,:))
%     [ordered, ordi] = sort(features(:,i));
%     figure()
%     scatter(ordered, target(ordi,:));
% end


% Conver to one-hot encoding for patternnet
% -------------------------------------
% ohY = zeros(length(Yo),10);
% for n = 1:length(Yo)
%     ohY(n,Yo(n)) = 1;
% end

%plot feature importance
% %----------------------
% [idx, scores] = fscmrmr(X,Y);
% bar(scores(idx));
% xlabel('Predictor rank');
% ylabel('Predictor importance');
% xticks([1:1:length(idx)]);
% xticklabels(X.Properties.VariableNames(idx));
% xtickangle(45);
% top_two = idx(1:2); %get te indices of the top 2 predictors
% 
%  X = X{:,:};
%  Y = Y{:,:};

for col = 1:3
    plot(N,t_perf(:,col));
    hold on
end
hold off
legend("traincgb","traincgf","traincgp");

% plot(N,t_time_old(1,:));
% hold on
% plot(N,t_time_old(2,:));
% plot(N,t_time_old(3,:));
% plot(N,t_time_old(4,:));
% legend("trainscg","trainrp","trainlm","trainbr");
% hold off



