% LOAD THE DATA
clear all;
load "data_batch_1.mat";
X(1:100,:) = data(1:100,:);
y(1:100,1) = labels(1:100,:);
%load "data_batch_2.mat";
%X(10001:20000,:) = data;
%y(10001:20000,1) = labels;
%load "data_batch_3.mat";
%X(20001:30000,:) = data;
%y(20001:30000,1) = labels;
%load "data_batch_4.mat";
%X(30001:40000,:) = data;
%y(30001:40000,1) = labels;
load "data_batch_5.mat";
X_test(1:10000,:) = data;
y_test(1:10000,1) = labels;

X = double(X);
y = double(y);
X_test = double(X_test);
y_test = double(y_test);



options = optimset('GradObj', 'on', 'MaxIter', 100);
initial_theta = zeros(size(X, 2), 1);

printf("TESTANDO LAMBDAS\n");
fflush(1);


%-------------------------------TESTAR LAMBDAS!!!!!!!!!!!!!!!!!!
lambda = 0.01;
sum_f1 = 0;
for i = [0:9]
  
  y_train = (y == i);
  y_test_in = (y_test == i);
 
  [theta, J, exit_flag] = fminunc(@(t)(costFunctionReg(t, X, y_train, lambda)), initial_theta, options);
  f1_score = f1(X_test, theta, y_test_in);
  sum_f1 += f1_score;
  
end

printf("f1 final de %f\n", sum_f1/10);
fflush(1);



lambda = 1;
sum_f1 = 0;
for i = [0:9]
  
  y_train = (y == i);
  y_test_in = (y_test == i);
  
  lambda = 0.01;
  [theta, J, exit_flag] = fminunc(@(t)(costFunctionReg(t, X, y_train, lambda)), initial_theta, options); 
  f1_score = f1(X_test, theta, y_test_in);
  sum_f1 += f1_score;
  
end

printf("f1 final de %f\n", sum_f1/10);
fflush(1);



lambda = 100;
sum_f1 = 0;
for i = [0:9]
  
  y_train = (y == i);
  y_test_in = (y_test == i);
  
  lambda = 0.01;
  [theta, J, exit_flag] = fminunc(@(t)(costFunctionReg(t, X, y_train, lambda)), initial_theta, options);  
  f1_score = f1(X_test, theta, y_test_in);
  sum_f1 += f1_score;
  
end

printf("f1 final de %f\n", sum_f1/10);
fflush(1);

%--------------FIM DO TESTAR LAMBDAS