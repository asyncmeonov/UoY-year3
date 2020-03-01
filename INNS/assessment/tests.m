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
%-------------------------------------
% ohY = zeros(length(Y.Variables),10);
% for n = 1:length(Y.Variables)
%     ohY(n,Y.CLASS(n)) = 1;
% end

%plot feature importance
%----------------------
[idx, scores] = fscmrmr(X,Y);
bar(scores(idx));
xlabel('Predictor rank');
ylabel('Predictor importance');
xticks([1:1:length(idx)]);
xticklabels(X.Properties.VariableNames(idx));
xtickangle(45);
top_two = idx(1:2); %get te indices of the top 2 predictors