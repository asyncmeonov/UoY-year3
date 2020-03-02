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
[Xo, Yo] = sm(X,Y);

function [Xo, Yo] = sm(X,Y)
    Xo = X;
    Yo = Y;
    [Ycount, Yr] = groupcounts(Y);
    for class = 1:10
%         [Xcount, Xr] = groupcounts(X);
        while Ycount(class) < max(Ycount)
            randRowId = randsample(find(Y == class), 1);
            dupXRow = X(randRowId,:);
            Xo = [Xo; dupXRow];
            dupYRow = Y(randRowId,:);
            Yo = [Yo; dupYRow];
            Ycount(class) = Ycount(class) + 1;
        end
    end
end