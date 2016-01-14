function net = init_lenet_2

rng('default');
rng(0) ;

f=1/100 ;

net = MateNet( {
  MateConvLayer(f*randn(5,5,1,32, 'single'), zeros(1, 32, 'single'), ...
                'stride', 1, 'pad', 0, 'name', 'conv1')
  MateReluLayer
  MatePoolLayer('pool',[2 2], 'stride', 2, 'pad', 0)
  MateConvLayer(f*randn(5,5,32,64, 'single'), zeros(1, 64, 'single'), ...
                'stride', 1, 'pad', 0, 'weightDecay', [0.005 0.005])
  MateReluLayer
  MatePoolLayer('pool',[2 2], 'stride', 2, 'pad', 0)  
  MateConvLayer(f*randn(4,4,64,500, 'single'), zeros(1, 500, 'single'), ...
                'stride', 1, 'pad', 0, 'weightDecay', [0.005 0.005])  
  MateReluLayer
  MateConvLayer(f*randn(1,1,500,1, 'single'), ...
                zeros(1, 1, 'single'), ...
                'weightDecay', [0.005 0.005], 'name','prediction')
  MateLogisticLossLayer('name','loss',...
                'takes',{'prediction','input:2'})
  MateLogisticErrorLayer('name','error',...
                'takes',{'prediction','input:2'})
} );