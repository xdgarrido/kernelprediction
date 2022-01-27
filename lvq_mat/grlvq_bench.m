%%% bench GRLVQ
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
load data/hist.txt; 
load data/quant8000.csv;
Prototypes = quant8000(:,:);
nb_epochs = 150;
%% run GRLVQ with in build optimization
GRLVQ_results = struct('GRLVQ_model',{},'GRLVQ_setting',{},'trainError',{},'testError',{});

GRLVQparams = struct('PrototypesPerClass',hist,'regularization',1);

[GRLVQ_model,GRLVQ_setting,trainError,testError,costs] = GRLVQ_train(trainSet, trainLab,'PrototypesPerClass',GRLVQparams.PrototypesPerClass,...
	'testSet',[testSet,testLab],'regularization',GRLVQparams.regularization, 'initialPrototypes', Prototypes, 'optimization', 'sgd','nb_epochs', 100, 'learningRatePrototypes', [0.00001, 0.0000005], 'learningRateRelevances', [1.0000e-03, 1.0000e-04]);

fprintf('GRLVQ: error on the train set: %f\n',trainError(end));

fprintf('GRLVQ: error on the test set: %f\n',testError(end));
%% plot relevances 
h1=figure(1); 
bar(GRLVQ_model.lambda);
box on;
title('GRLVQ: relevances');
print(h1,'lambdas','-dpng');

%% csv files 
final_model = [GRLVQ_model.w, GRLVQ_model.c_w];
lambdas = GRLVQ_model.lambda;
csvwrite("quant_opt.csv", final_model);
csvwrite("lambdas.csv", lambdas);

%% plot accuracy and cost function
x= 0:nb_epochs;
h2=figure(2); 
plot(x,costs); 
xlabel('epochs'); 
ylabel('cost'); 
title('GRLVQ training'); 
print(h2,'cost','-dpng');

h3=figure(3); 
rect = [0.50, 0.50, .25, .25]; 
h=plot(x,100. * (1. - trainError),x,100. * (1. -testError)); 
xlabel('epochs'); 
ylabel('accuracy (%)'); 
legend('training set accuracy','test set accuracy','Location',rect); 
title('single predictor GRLVQ'); 
print(h3,'accuracy','-dpng');

