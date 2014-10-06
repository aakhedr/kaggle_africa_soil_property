clear; close all; clc;

data = csvread('training.csv', 1, 1);
% ==============================================================================
% 60% of 1157 observations ~= 695 examples (training set)
% ==============================================================================
X = data(1:695, 1:3594); y = data(1:695, 3596);		% P label only

[m, n] = size(X);
% add intercept term
X = [ones(m, 1) X];

% ==============================================================================
% 20% of 1157 observations ~= 231 examples (Cross validation set)
% ==============================================================================
Xval = data(696:926, 1:3594); yval = data(696:926, 3596);	% P label only
% Add intercept
Xval = [ones(size(Xval, 1), 1) Xval];

% ==============================================================================
% Neural Networks 
% ==============================================================================
input_layer_size  = 3594;			% Number of features
hidden_layer_size = 250;			% The more the better
num_labels = 1;						% P prediction

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

% ==============================================================================
% Training the neural network using fmincg 
% ==============================================================================
lambda = 0.03;
nn_params = trainNN(initial_nn_params, input_layer_size, hidden_layer_size, ...
	num_labels, X, y, lambda);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% ==============================================================================
% 20% of 1157 observations ~= 231 examples (Test set)
% ==============================================================================
Xtest = data(927:end, 1:3594); ytest = data(927:end, 3596);	% P label only
% Add intercept
Xtest = [ones(size(Xtest, 1), 1) Xtest];

% =============================================================================
% Using learned parameters to make predictions on the test set
% ==============================================================================
h1 = Xtest * Theta1';
h2 = [ones(size(h1, 1), 1) h1] * Theta2';

SSE = sum((h2 - ytest).^2);
fprintf('Sum of squared erros on the test set (231 examples) is: %f\n', SSE);

% ==============================================================================
% ==============================================================================
% ==============================================================================
