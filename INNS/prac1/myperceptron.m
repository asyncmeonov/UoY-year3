% THIS IS WRITTEN AS A FUNCTION
% REMOVE THE DECLARATION LINE BELOW TO USE IT AS A SCRIPT

function [iterations, weights] = myperceptron(input_matrix, output_class, max_iterations)

% input_matrix - matrix containing data, variables as rows and observations
%                   as columns
% output_class - vector (1 x n matrix) of class labels {-1,+1}
% max_iterations - maximum number of iterations you set for the algorithm
% iterations - returns number of iterations
% weight - return the weight vector, including bias weight

% You could set a learning rate - comment out if you want to keep it simple
eta = 0.1;

% Get the number of data points N and number of variables V
N = size(input_matrix,2); % number of columsn = data points
V = size(input_matrix,1); % number of rows = input variables

% The Perceptron needs a bias input as well as the data inputs
% To do this we add a row of 1's as the first row of the data matrix 
% - this is a fixed bias input of 1 for every observation
inputs_and_bias = [ ones(1,N) ; input_matrix ];

% We then create the weight vector and give it random values.
% Remember there are V inputs and a bias, so there are V+1 weights 
weights = rand( V+1, 1 ); 

% To control learning, set a flag for whether data is correctly classified 
% Initialise it to 'false' so we take at least the first training step
all_correct = false; 

% We will calculate the output from the neuron as the weighted sum of its
% inputs, and threshold it to give a class label of {-1,+1}.
% We initialise the vector to record the calculated class with zeros.
calculated_class = zeros( 1, N );

% We initialise a counter to keep track of the number of training
% iterations
iterations = 0;

% To train the perceptron, go round the training loop until it converges 
% on a correct classifier or we reach the maximum number of iterations
while (all_correct == false) && (iterations < max_iterations)
    
    iterations = iterations + 1;
    
    % check whether patterns are all classified correctly
    all_correct = true; % assume they are to start with...
    
    %% INSERT YOUR CODE HERE TO CALCULATE THE CLASS FOR ALL OBSERVATIONS
    % This is the calculation of class for every observation (column) 
    % given the current weights. Remember calculated_class is a vector with
    % one element for each observation.
    y = dot(weigths,input_matrix)
    for entry = 1:N
        if y(entry) ==  input_matrix(3:entry)
            continue;
        elseif y(entry) == 1
            weights(entry) = weights(entry) - eta*input_matrix(1,entry); 
        else
            weights(entry) = weights(entry) + eta*input_matrix(1, entry);
        end
    end
    
    
    calculated_class; % calculate it from input_matrix and weights

    % compare calculated class and desired class for every observation   
    if any(calculated_class ~= output_class) % if any don't match...
        all_correct = false; % set the flag to false 
    end

    % If the flag is false (some observation is classified incorrectly), 
    % adjust the weights
    if all_correct == false
        %% INSERT YOUR CODE HERE TO ADJUST THE WEIGHTS        
        % You need to modify the weights based on the misclassified data
        % Most straightforward is to look at each point sequentially,
        % checking whether it is misclassified. if it is, adjust the
        % weights
        

        % if the data point is misclassified, calculate a weight
        % change from the input data, the desired output class and
        % the learning rate if you are using one explicitly.
        
    end
end
