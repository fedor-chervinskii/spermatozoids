function [bi_net,info,dataset] = MateOrientationBinaryClassifier()
%example is derived from the analogous MatConvNet example

m = 28

data = CollectPatchesWithRotations('labels/train/','images/train/', m, 2, true);
data = data(data(:, end) >= 0, :); %only positive patches
data = data(randperm(size(data, 1)), :)';
n = size(data,2);

angles = data(end,:);
data = data(1:end-1,:);

fprintf('number of samples after augmentation %d\n',n);

dataset = struct;
dataset.imdb.images.angles = angles;
dataset.imdb.images.labels = round(angles./180 + 0);
dataset.imdb.images.data = single(reshape(data,m,m,1,[])) ;
%dataset.imdb.images.data = single(reshape(data,m,m,1,[])) ;

dataset.imdb.images.data = dataset.imdb.images.data - 122;

n = size(dataset.imdb.images.angles,2);

dataset.imdb.images.set = [ones(1,n - 500) 3*ones(1, 500)] ;
dataset.imdb.meta.sets = {'train', 'val', 'test'} ;

[thispath,~,~] = fileparts(mfilename('fullpath'));
expDir = [thispath '/exp'] ;
useGpu = false;

% Define the network

f=1/100 ;

net = MateNet( {
  MateFlattenLayer()
  MateFullLayer(f*randn(400, 784, 'single'), zeros(400, 1, 'single'))
  MateReluLayer
  MateFullLayer(f*randn(1, 400, 'single'), zeros(1, 1, 'single'),... 
                'name', 'prediction')
  MateLogisticLossLayer('name','loss',...
                'takes',{'prediction','input:2'})
  } );


%subtract mean
dataset.imdb.images.data = bsxfun(@minus, dataset.imdb.images.data,...
            mean(dataset.imdb.images.data,4)) ;
          
%move to GPU         
if useGpu
  net = net.move('gpu');
  dataset.imdb.images.data = gpuArray(dataset.imdb.images.data) ;
  dataset.imdb.images.labels = gpuArray(dataset.imdb.images.labels) ;
end

dataset.train = find(dataset.imdb.images.set == 1);
dataset.val = find(dataset.imdb.images.set == 3);
dataset.batchSize = 100;

[net,info,dataset] = net.trainNet(@getBatch, dataset,...
     'numEpochs',20, 'continue', false, 'expDir', expDir,...
     'learningRate', 0.01, 'monitor', {'loss'},...
     'onEpochEnd', @onEpochEnd) ;

bi_net = net;
save('exp/bi_net.mat', 'bi_net');

%----------------------------------------------------------%

function [x, eoe, dataset] = getBatch(istrain, batchNo, dataset)
eoe=false;
batchStart = (batchNo-1)*dataset.batchSize+1;
batchEnd = batchNo*dataset.batchSize;

if istrain
  if batchEnd >= numel(dataset.train)
    batchEnd = numel(dataset.train);
    eoe = true; %end of epoch
  end
  batch = dataset.train(batchStart:batchEnd);
else
  if batchEnd >= numel(dataset.val)
    batchEnd = numel(dataset.val);
    eoe = true; %end of epoch
  end
  batch = dataset.val(batchStart:batchEnd); 
end

x{1} = dataset.imdb.images.data(:,:,:,batch) ;
labels = dataset.imdb.images.labels(batch) ;
x{2} = zeros([1 numel(batch)],'single');
x{2}(1,:) = labels(:)*2 - 1;

function [net,dataset,learningRate] = onEpochEnd(net,dataset,learningRate)
1;
%learningRate = 0.9 * learningRate;