function [history, stoppedGettingBetter, theta] = linear (X, y, theta)

learnRate = 0.01;
oldcost = 0;
iteractions = 100;
k = 1;
m = size(y, 1);
for j = [1:iteractions]
	pred = X * theta';
	theta = theta - learnRate * (1/m) * (pred - y)' * X;
	cost = costFunction(X, y, theta);
	if(oldcost == cost)	%when did the model stop to get better?
		stoppedGettingBetter(k++) = j;
	endif
	oldcost = cost;
	history(j) = cost;	%store the historic of costs to plot later
end