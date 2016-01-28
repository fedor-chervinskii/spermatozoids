function net = init_bi_net

rng('default');
rng(0) ;

f=1/100 ;

% net = MateNet( {
%   MateConvLayer(regr_net.layers{1,1}.weights.w{1,1}, ...
%                 regr_net.layers{1,1}.weights.w{1,2}, ...
%                 'stride', 1, 'pad', 0, 'name', 'conv1')
%   MatePoolLayer('pool',[2 2], 'stride', 2, 'pad', 0)
%   MateConvLayer(f*randn(5,5,20,50, 'single'), zeros(1, 50, 'single'), ...
%                 'stride', 1, 'pad', 0, 'weightDecay', [0.005 0.005])
%   MatePoolLayer('pool',[2 2], 'stride', 2, 'pad', 0)  
%   MateConvLayer(f*randn(4,4,50,50, 'single'), zeros(1, 50, 'single'), ...
%                 'stride', 1, 'pad', 0, 'weightDecay', [0.005 0.005])  
%   MateReluLayer
%   MateConvLayer(f*randn(1,1,50,1, 'single'), zeros(1, 1, 'single'),... 
%                 'weightDecay', [0.005 0.005],'name','prediction')
% 
%   MateFlattenLayer()
%   MateFullLayer(f*randn(500, 784, 'single'), zeros(500, 1, 'single'))
%   MateReluLayer
%   MateFullLayer(f*randn(1, 500, 'single'), zeros(1, 1, 'single'),... 
%                 'name', 'prediction')
%   MateLogisticLossLayer('name','loss',...
%                 'takes',{'prediction','input:2'})
%   MateLogisticErrorLayer('name','error',...
%                 'takes',{'prediction','input:2'})
% } );

net = MateNet( {
  MateConvLayer(f*randn(5,5,1,20, 'single'), zeros(1, 20, 'single'), ...
                'stride', 1, 'pad', 0, 'name', 'conv1')
  MatePoolLayer('pool',[2 2], 'stride', 2, 'pad', 0)
  MateConvLayer(f*randn(5,5,20,50, 'single'), zeros(1, 50, 'single'), ...
                'stride', 1, 'pad', 0, 'weightDecay', [0.005 0.005])
  MatePoolLayer('pool',[2 2], 'stride', 2, 'pad', 0)  
  MateConvLayer(f*randn(4,4,50,500, 'single'), zeros(1, 500, 'single'), ...
                'stride', 1, 'pad', 0, 'weightDecay', [0.005 0.005])  
  MateFlattenLayer
  MateFullLayer(f*randn(2, 500, 'single'), zeros(2, 1, ...
                'single'), 'weightDecay', [0.005 0.005], 'name','prediction')
  MateSoftmaxLossLayer('name','loss',...
                'takes',{'prediction','input:2'})
  MateSoftmaxLayer('name','softmax','takes','prediction','skipBackward',1)
  MateMultilabelErrorLayer('name','error',...
                'takes',{'prediction','input:2'})
  } );
