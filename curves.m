clear; close all; clc;

data = csvread('training.csv', 1, 1);

X = data(1:800, 1:3593); y = data(1:800, 3595);		% Ca label only

[m, n] = size(X);
% add intercept term
X = [ones(m, 1) X];

Xval = data(811:end, 1:3593); yval = data(811:end, 3595);	% Ca label only
% Add intercept
Xval = [ones(size(Xval, 1), 1) Xval];

% ==============================================================================
% Neural Networks 
% ==============================================================================
input_layer_size  = 3593;			% Number of features
hidden_layer_size = 100;			% The more the better
num_labels = 1;						% SAND prediction

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, hidden_layer_size);
initial_Theta3 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:); initial_Theta2(:); initial_Theta3(:)];

% % ==============================================================================
% % Learning curve (Neural Networks 2 hidden layers)
% % ==============================================================================
% lambda = 0;
% [error_train, error_val] = learningCurve_NN(X, y, Xval, yval, ...
% 	lambda, initial_nn_params, input_layer_size, hidden_layer_size, num_labels)

% plot(1:m, error_train, 1:m, error_val);
% title('Learning curve for Neural Networks 2 hidden layers - 100 Neurons')
% legend('Train', 'Cross Validation')
% xlabel('Number of training examples on SAND label only')
% ylabel('Error')

% fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
% for i = 1:m
%     fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
% end


% ==============================================================================
% Validation curve to choose the best lambda value (Neural Networks)
% ==============================================================================
[lambda_vec, error_train, error_val] = validationCurve_NN(X, y, Xval, yval, ...
	initial_nn_params, input_layer_size, hidden_layer_size, num_labels)

plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');

fprintf('lambda\t\tTrain Error\tValidation Error\n');
for i = 1:length(lambda_vec)
	fprintf(' %f\t%f\t%f\n', lambda_vec(i), error_train(i), error_val(i));
end

	% SAND lambda = .003 - 2 hidden layers 100 Neurons
	% SOC lambda = .0003 - 2 hidden layers 100 Neurons
	% pH lambda = .1 - 2 hidden layers 100 Neurons
	% P lambda = 10 - 2 hidden layers 100 Neurons
	% Ca lambda = .03 - 2 hidden layers 100 Neurons
% ==============================================================================
