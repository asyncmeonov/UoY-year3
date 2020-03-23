classdef preprocess
    %PREPROCESS Utiliti functions for data transformation
    
    properties
      %  void properties
    end
    
    methods(Static)
        function [trainInd,valInd,testInd] = divideblock(X)
            [trainInd,valInd,testInd] = divideblock(length(X), 0.7, 0.15,0.15);
        end
        function [ohY] = one_hot_encode(Y)
            ohY = zeros(length(Y),10);
            for n = 1:length(Y)
                ohY(n,Y(n)) = 1;
            end
        end
        function [Xr, Yr] = remove_outliers(X,Y)
            [Xr, rId] = rmoutliers(X);
            Yr = Y(~rId);
            
        end
        function [X] = standardise(X)
            % maps to 0 mean, unit variance space
            X = mapstd(X);
        end
        function [X] = normalise(X)
            % maps to [-1;1] space
            X = mapminmax(X);
        end
        function [idX] = feature_importance(X,Y)
            [idX, scores] = fscmrmr(X,Y); %can be used under R2019b but not a
            bar(scores(idx));
            xlabel('Predictor rank');
            ylabel('Predictor importance');
            xticks([1:1:length(idX)]);
            xticklabels(X.Properties.VariableNames(idX));
            xtickangle(45);
        end
        

        function [Xo, Yo] = oversample(X,Y)
            %OVERSAMPLE handle unbalanced data
            %find the most occurent class and multiply all others until they reach him
            Xo = X;
            Yo = Y;
            [Ycount, Yr] = groupcounts(Y);
            for class = 1:numel(unique(Y))
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
        
        function [Xo, Yo] = scale_to_mean(X,Y)
            %TODO
        end
%         function outputArg = method1(obj,inputArg)
%             %METHOD1 Summary of this method goes here
%             %   Detailed explanation goes here
%             outputArg = obj.Property1 + inputArg;
%         end
    end
end

