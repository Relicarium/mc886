data_train = load("-ascii", "year-prediction-msd-train.txt");
y_train = data_train(:,1);
x_train = data_train(:,2:91);
x_train = [ones(size(x_train, 1), 1) x_train];
whos
data_test = load("-ascii", "year-prediction-msd-test.txt");
y_test = data_test(:,1);
x_test = data_test(:,2:91);
x_test = [ones(size(x_test, 1), 1) x_test];
for i = [2:size(x_train,2)]
  x_train(:,i) = (x_train(:,i) - mean(x_train(:,i)))/std(x_train(:,i));
  x_test(:,i) = (x_test(:,i) - mean(x_test(:,i)))/std(x_test(:,i));
end
whos