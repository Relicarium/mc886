  function [history, theta] = linear (X, y, theta)

learnRate = 0.001;
iteractions = 5000;
k = 1;
m = size(y, 1);
size(X);
size(theta);
size(y);
y(1,1);
for j = [1:iteractions]
	pred = theta * X';
	theta = theta - learnRate * (1/m) .* (pred - y') * X;
	cost = costFunction(X, y, theta)
	history(j) = cost;	%store the historic of costs to plot later
end