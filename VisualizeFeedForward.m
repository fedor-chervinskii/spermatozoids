function [ output_param ] = VisualizeFeedForward 

[net,info,dataset] = MateDetection;
patches = MakeTestPatches(29);
net = net.makePass({single(patches); single(zeros(2, 10))});

end