% Initialization
clear, close all; clc

% read data into MATLAB
% Topsoil=1 & Subsoil=0
fprintf('Loading training.csv data into MATLAB...\n')
data = csvread('training.csv', 1, 1);

fprintf('Dividing data into train and test sets...\n\n')
% train data = 70% of total (~= 800 rows)
train = data(1:800, :);
x = train(:, 1:3594); y = train(:, 3595:end);
[m, n] = size(x);

% add intercept term to x
X = [ones(m, 1) x];
%=================================================================

fprintf('Solving with normal equation...\n')
% calculate the parameters from the normal equation
theta = normalEqn(X, y);
fprintf('Calculated the parameters using the normal equation.\n\n');
%=================================================================
%% Running parameters on the test set %%
fprintf('Trying the parameters learned on the test set...\n\n');
test = data(801:end, :);
xtest = test(:, 1:3594); ytest = test(:, 3595:end);
[l, k] = size(xtest);
Xtest = [ones(l, 1) xtest];

predictTest = Xtest * theta;
% calculate SSE
SSE = sum((predictTest - ytest).^2);

totalErrors = sqrt(SSE);
fprintf('Total errors per column on 357 examples (test set) are:\n');
disp(totalErrors);
%========================================================================
%% Running parameters on the training set too %%
fprintf('Total errors per column on 800 examples (training set) are:\n');
predict = X * theta;
errors = sqrt(sum(predict - y).^2);
disp(errors);
%==========================================================================

%% Read Test Data into MATLAB %%
% Topsoil=1 & Subsoil=0
fprintf('Loading training.csv data into MATLAB...\n')
testData = csvread('sorted_test.csv', 1, 1);

% add bias 1s
testData = [ones(size(testData, 1), 1) testData];
predictions = testData * theta;

% write data to a csv file
fprintf('Write predictions to submission1.csv file\n\n')
csvwrite('submission1.csv', predictions);
%===============================================================================
