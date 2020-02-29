features = CTG{:,:};
target = Y{:,:};
for i = 1:length(features(1,:))
    [ordered, ordi] = sort(features(:,i));
    figure()
    scatter(ordered, target(ordi,:));
end


% Conver to one-hot encoding for patternnet
% ohY = zeros(length(Y.Variables),10);
% for n = 1:length(Y.Variables)
%     ohY(n,Y.CLASS(n)) = 1;
% end