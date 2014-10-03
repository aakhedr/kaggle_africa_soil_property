clear; close all, clc;

data = csvread('training.csv', 1, 1);
%==============================================================================
% 60% of 1157 observations ~= 695 examples (training set)
%==============================================================================
X = data(1:695, 1:3594); y = data(1:695, 3595);		% Ca label only

[m, n] = size(X);
% add intercept term
X = [ones(m, 1) X];
%==============================================================================
% Cross validation set (20% of 1157 observations ~= 231 examples)
%==============================================================================
Xval = data(696:925, 1:3594); yval = data(696:925, 3595);	% Ca label only
% Add intercept
Xval = [ones(size(Xval, 1), 1) Xval];

%==============================================================================
% Learning curve %
%==============================================================================
% lambda = 0;
% [error_train, error_val] = learningCurve(X, y, Xval, yval, lambda);

% plot(1:m, error_train, 1:m, error_val);
% title('Learning curve for linear regression')
% legend('Train', 'Cross Validation')
% xlabel('Number of training examples')
% ylabel('Error')

% fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
% for i = 1:m
%     fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
% end

% fprintf('Program paused. Press enter to continue.\n');
% pause;
%==============================================================================
% Neural Networks 
%==============================================================================
input_layer_size  = 3594;			% Number of features
hidden_layer_size = 5000;			% The more the better
num_labels = 1;						% Ca prediction

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

lambda = 0;

[J, grad] = nnCostFunction(initial_nn_params, input_layer_size, ...
	hidden_layer_size, num_labels, X, y, lambda);






%==============================================================================
% Validation curve to choose the best lambda value
%==============================================================================
% [lambda_vec, error_train, error_val] = validationCurve(X, y, Xval, yval)

% plot(lambda_vec, error_train, lambda_vec, error_val);
% legend('Train', 'Cross Validation');
% xlabel('lambda');
% ylabel('Error');

% fprintf('lambda\t\tTrain Error\tValidation Error\n');
% for i = 1:length(lambda_vec)
% 	fprintf(' %f\t%f\t%f\n', lambda_vec(i), error_train(i), error_val(i));
% end

% fprintf('Program paused. Press enter to continue.\n');
% pause;
%==============================================================================