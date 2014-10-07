clear; close all; clc;

data = csvread('training.csv', 1, 1);

% 60% of 1157 observations ~= 695 examples (training set)
% ==============================================================================
X = data(1:695, 1:3594); y = data(1:695, 3596);		% P label only

[m, n] = size(X);
% add intercept term
X = [ones(m, 1) X];

% 20% of 1157 observations ~= 231 examples (Cross validation set)
% ==============================================================================
Xval = data(696:926, 1:3594); yval = data(696:926, 3596);	% P label only
% Add intercept
Xval = [ones(size(Xval, 1), 1) Xval];

