function [bi_net,info,dataset] = train_orient_bin_classifier()
%example is derived from the analogous MatConvNet example

expDir = fullfile('exp','bi_net');
dataDir = fullfile('exp');
imdbPath = fullfile(dataDir, 'imdb_bi.mat');

dataset = struct;

dataset.imdb = load(imdbPath);

dataset.train = find(dataset.imdb.images.set == 1);
dataset.val = find(dataset.imdb.images.set == 3);
dataset.batchSize = 100;

% Define the network

net = init_bi_net();

[net,info,dataset] = net.trainNet(@getBatch, dataset,...
     'numEpochs',25, 'continue', true, 'expDir', expDir,...
     'learningRate', 0.0005, 'monitor', {'loss','error'},...
     'showLayers', 'conv1',...
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

x{1} = single(dataset.imdb.images.data(:,:,:,batch)) ;
labels = dataset.imdb.images.angles(batch)./180 ;
%x{2} = zeros([1 numel(batch)],'single');
%x{2}(1,:) = labels(:)*2 - 1;
x{2} = zeros([2 numel(batch)],'single');
x{2}(sub2ind([2 numel(batch)],labels(:) + 1,(1:numel(batch))')) = single(1);

function [net,dataset,learningRate] = onEpochEnd(net,dataset,learningRate)
1;
%learningRate = 0.9 * learningRate;