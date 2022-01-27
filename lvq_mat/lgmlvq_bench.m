%%% bench LGMLVQ
% needs files: normalized training set (ts.csv)
%              normalized test set (cs.csv)
%              classifier text file (quant6000.txt)
%              histogram file (number of points mapped to a particular
%              class)
%  all these files are obtained from filtercluster program
%
addpath(genpath('.'));
%load the data
% actData = 'Iris';
load data/ts.csv;
trainSet = ts(:,1:end-1);
trainLab = ts(:,end);
load data/cs_norm.csv 
testSet = cs_norm(:,1:end-1);
testLab = cs_norm(:,end);
actData = 'KernePred';
load data/hist8000.txt; 
load data/quant8000.csv;
Prototypes = quant8000(:,:);
nb_epochs = 60;
%% run LGMLVQ with in build optimization
LGMLVQ_results = struct('LGMLVQ_model',{},'LGMLVQ_setting',{},'trainError',{},'testError',{});
projectionDimension = size(trainSet,2);
LGMLVQparams = struct('PrototypesPerClass',hist8000,'regularization',1);

[LGMLVQ_model,LGMLVQ_setting,trainError,testError,costs] = LGMLVQ_train(trainSet, trainLab,'PrototypesPerClass',LGMLVQparams.PrototypesPerClass,...
	'testSet',[testSet,testLab],'regularization',LGMLVQparams.regularization, 'initialPrototypes', Prototypes, 'optimization', 'sgd','nb_epochs', nb_epochs, 'learningRatePrototypes', [0.0001, 0.0000005],'learningRateMatrix', [1.0000e-03, 1.0000e-05]);

fprintf('LGMLVQ: error on the train set: %f\n',trainError(end));

fprintf('LGMLVQ: error on the test set: %f\n',testError(end));
%% plot relevances 


%% csv files 
final_model = [LGMLVQ_model.w, LGMLVQ_model.c_w];
psis= LGMLVQ_model.psis;
csvwrite("quantp_opt.csv", final_model);
csvwrite("psis.csv", psis);

%% plot accuracy and cost function
x= 0:nb_epochs;
h2=figure(2); 
plot(x,costs); 
xlabel('epochs'); 
ylabel('cost'); 
title('LGMLVQ training'); 
print(h2,'costo','-dpng');

h3=figure(3); 
rect = [0.50, 0.50, .25, .25]; 
h=plot(x,100. * (1. - trainError),x,100. * (1. -testError)); 
xlabel('epochs'); 
ylabel('accuracy (%)'); 
legend('training set accuracy','test set accuracy','Location',rect); 
title('single predictor LGMLVQ'); 
print(h3,'accuracyo','-dpng');

