%calcula custo
pred = funcaodepredicao[X, theta]
cost = (1/(2*sizeData)) * sum((pred - y).**2)


%%atualiza
k = 0
for j = [1:iteractions]
	pred = funcaodepredicao[X, theta]
	theta = theta - learnRate * (1/m) * (pred - y)' * X
	cost = costFunction(parameters, answers, theta)
	if(oldcost == cost)	%when did the model stop to get better?
		stoppedGettingBetter(k++) = j
	endif
	oldcost = cost
	history(j) = cost	%store the historic of costs to plot later
end
