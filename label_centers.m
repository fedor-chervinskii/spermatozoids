name = 'train/C001H001S0001000002_1'
figure, imshow(['images/' name '.tif']);
f = fopen(['labels/centers/' name '.csv'],'a+');
centers = load(['labels/centers/' name '.csv']);
hold on;
plot(centers(:,1),centers(:,2),'.r');
frewind(f)
while 1
    h = impoint;
    position = wait(h);
    fprintf(f,'%.2f,%.2f\n',position);
end