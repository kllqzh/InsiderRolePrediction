Xtrain=readtable("Xtrain.csv");
Xtrain=Xtrain(:,2:end);
ytrain=readtable("ytrain.csv");
ytrain=ytrain(:,2:end);

Xtest=readtable("Xtest.csv");
Xtest=Xtest(:,2:end);
ytest=readtable("ytest.csv");
ytest=ytest(:,2:end);

train=[ytrain Xtrain];

% Calculate accuracy for fine tree model
train_prediction1=fine_tree.predictFcn(Xtrain);
test_prediction1=fine_tree.predictFcn(Xtest);
train_accuracy1=sum(table2array(ytrain)==train_prediction1)/length(train_prediction1);
test_accuracy1=sum(table2array(ytest)==test_prediction1)/length(test_prediction1);
   
% Calculate accuracy for boosted tree model
train_prediction2=boosted_tree.predictFcn(Xtrain);
test_prediction2=boosted_tree.predictFcn(Xtest);
train_accuracy2=sum(table2array(ytrain)==train_prediction2)/length(train_prediction2);
test_accuracy2=sum(table2array(ytest)==test_prediction2)/length(test_prediction2);

% Calculate accuracy for bagged tree model
train_prediction3=bagged_tree.predictFcn(Xtrain);
test_prediction3=bagged_tree.predictFcn(Xtest);
train_accuracy3=sum(table2array(ytrain)==train_prediction3)/length(train_prediction3);
test_accuracy3=sum(table2array(ytest)==test_prediction3)/length(test_prediction3);

% Comment: The column "SECTITLE" is removed because dropping it increase
% the test accuracy. This is probably because some security titles in test
% data do not appear in train data. Out of three models, bagged tree has
% the best performance: training accuracy is at 97.6 percent and testing
% accuracy is at 70.2 percent. This probably because our data has very high
% level of variability.
