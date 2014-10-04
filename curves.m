% ==============================================================================
% Learning curve (Linear Regression)
% ==============================================================================
lambda = 0;
[error_train, error_val] = learningCurve(X, y, Xval, yval, lambda);

plot(1:m, error_train, 1:m, error_val);
title('Learning curve for linear regression')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')

fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:m
    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
end

% ==============================================================================
% Validation curve to choose the best lambda value (Linear Regression)
% ==============================================================================
[lambda_vec, error_train, error_val] = validationCurve_LR(X, y, Xval, yval)

plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');

fprintf('lambda\t\tTrain Error\tValidation Error\n');
for i = 1:length(lambda_vec)
	fprintf(' %f\t%f\t%f\n', lambda_vec(i), error_train(i), error_val(i));
end

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

		% ACCORDING TO validationCurve_NN.jpg => BEST LAMBDA = .1 %%
% ==============================================================================
