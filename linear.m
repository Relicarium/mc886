  function [history, stoppedGettingBetter, theta] = linear (X, y, theta)

learnRate = 1 * 10e-08;
iteractions = 30000;
k = 1;
m = size(y, 1)
size(X)
size(theta)
size(y)
y(1,1)
for j = [1:iteractions]
	pred = X * theta';
	theta = theta - learnRate * (1/m) .* (pred - y)' * X;
	cost = costFunction(X, y, theta)
	history(j) = cost;	%store the historic of costs to plot later
end