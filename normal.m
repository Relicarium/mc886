function [cost] = normal(X, y)

theta = (pinv(X'*X))*X'*y;

cost = costFunction(X, y, theta)