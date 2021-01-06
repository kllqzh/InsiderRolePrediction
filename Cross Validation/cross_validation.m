Xtrain=readtable("Xtrain.csv");
Xtrain=Xtrain(:,2:end);
ytrain=readtable("ytrain.csv");
ytrain=ytrain(:,2:end);
Xtest=readtable("Xtest.csv");
Xtest=Xtest(:,2:end);
ytest=readtable("ytest.csv");
ytest=ytest(:,2:end);
train=[ytrain Xtrain];
folds = 20;
indices = crossvalind('Kfold',numel(ytrain),folds);
x = [];
y = [];
z = [];
for i = 1:folds
    test = (indices == i);
    train_k = ~test;
    [trainedClassifier, validationAccuracy] = train_bagged_tree(train(train_k,:));
    btree = trainedClassifier.predictFcn(Xtrain(test,:));
    cv_accuracy = sum(table2array(ytrain(test,:))==btree)/length(btree);
    x = [x, cv_accuracy];
    disp(cv_accuracy);
end
for i = 1:folds
    test = (indices == i);
    train_k = ~test;
    [trainedClassifier, validationAccuracy] = train_boosted_tree(train(train_k,:));
    botree = trainedClassifier.predictFcn(Xtrain(test,:));
    cv_accuracy = sum(table2array(ytrain(test,:))==botree)/length(botree);
    y = [y, cv_accuracy];
    disp(cv_accuracy);
end
for i = 1:folds
    test = (indices == i);
    train_k = ~test;
    [trainedClassifier, validationAccuracy] = train_fine_tree(train(train_k,:));
    ftree = trainedClassifier.predictFcn(Xtrain(test,:));
    cv_accuracy = sum(table2array(ytrain(test,:))==ftree)/length(ftree);
    z = [z, cv_accuracy];
    disp(cv_accuracy);
end