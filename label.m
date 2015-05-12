name = 'C001H001S0001000001_3'
figure, imshow(['images/' name '.tif']);
f = fopen(['labels/' name '.csv'],'a+');
frewind(f)
tline = fgetl(f);
while ischar(tline)
    A = sscanf(tline, '%f,%f;%f,%f');
    line([A(1) A(2)],[A(3) A(4)]);
    tline = fgetl(f);
end
frewind(f)
while 1
    h = imline;
    position = wait(h);
    fprintf(f,'%.2f,%.2f;%.2f,%.2f\n',position);
end