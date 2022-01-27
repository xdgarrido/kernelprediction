%%% bench GMLVQ
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
load('data/ts.csv','ts');
trainSet = ts(:,1:end-1);
trainLab = ts(:,end);
load('data/cs_norm.csv','cs_norm'); 
testSet = cs_norm(:,1:end-1);
testLab = cs_norm(:,end);
actData = 'KernePred';
load ('data/hist8000.txt','hist8000'); 
load ('data/quant8000.csv','quant8000');
Prototypes = quant8000(:,:);
nb_epochs = 100;
%% run GMLVQ with in build optimization
GMLVQ_results = struct('GMLVQ_model',{},'GMLVQ_setting',{},'trainError',{},'testError',{});

GMLVQparams = struct('PrototypesPerClass',hist8000,'regularization',1.0);

%[GMLVQ_model,GMLVQ_setting,trainError,testError, costs] = traingmlvq(trainSet, trainLab,'PrototypesPerClass',GMLVQparams.PrototypesPerClass,...
%	'testSet',[testSet,testLab],'regularization',GMLVQparams.regularization, 'initialPrototypes', Prototypes, 'optimization','sgd', 'initialMatrix',diag([1 1 1 1 1]),'nb_epochs', nb_epochs, 'learningRatePrototypes', [0.75e-4, 1.e-6],'learningRateMatrix', [0.75e-4, 1.e-6]);
                
[GMLVQ_model,GMLVQ_setting,trainError,testError,costs] = traingmlvq(trainSet, trainLab,'PrototypesPerClass',GMLVQparams.PrototypesPerClass,...
	'testSet',[testSet,testLab],'regularization',GMLVQparams.regularization, 'initialPrototypes', Prototypes, 'optimization','sgd', 'initialMatrix',diag([1 1 1 1 1]),'nb_epochs', nb_epochs, 'learningRatePrototypes', [1e-4, 0.6, 4],'learningRateMatrix', [1.e-5, 0.6, 4]);
                
fprintf('GMLVQ: error on the train set: %f\n',trainError(end));

fprintf('GMLVQ: error on the test set: %f\n',testError(end));
%% plot relevances 


%% csv files 
final_model = [GMLVQ_model.w, GMLVQ_model.c_w];
omega = GMLVQ_model.omega;
csvwrite("quant8000_opt.csv", final_model);
csvwrite("omega8000.csv", omega);

%% plot accuracy and cost function
x= 0:nb_epochs;
h2=figure(2); 
plot(x,costs); 
xlabel('epochs'); 
ylabel('cost'); 
title('GMLVQ training'); 
print(h2,'costo_8000','-dpng');

h3=figure(3); 
rect = [0.50, 0.50, .25, .25]; 
h=plot(x,100. * (1. - trainError),x,100. * (1. -testError)); 
xlabel('epochs'); 
ylabel('accuracy (%)'); 
legend('training set accuracy','test set accuracy','Location',rect); 
title('single predictor GMLVQ'); 
print(h3,'accuracyo_8000','-dpng');

csvwrite("cost_8000.csv", costs);
csvwrite("accuracy_8000.csv", 100. * (1. -testError));
