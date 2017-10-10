function [J, grad] = costFunctionReg(theta, X, y, lambda)

m = length(y);

J = 0;
grad = zeros(size(theta));


h = sigmoid(X * theta);


J = (1/m)  * sum((-y.*log(h)) - (1 - y).*log(1-h)) + ((lambda/(2 *m)) * sum(realpow(theta(2:end),2))); 



theta_reg = theta;
theta_reg(1) = 0;
grad = 1/m * (X' * (h - y) + (lambda * theta_reg));

end