labels_dir = '/Users/fedor/research/Deep Learning/spermatozoids/labels/'
fileList = dir([labels_dir 'orientations/train/*.csv']);
fileList = {fileList.name}'
for i = 1:numel(fileList)
    f1 = fopen([labels_dir 'orientations/train/' fileList{i}],'r');
    f2 = fopen([labels_dir 'centers/train/' fileList{i}], 'w');
    tline = fgetl(f1);
    while ischar(tline)
        A = sscanf(tline, '%f,%f;%f,%f');
        point = [A(1) A(3)];
        fprintf(f2,'%.2f,%.2f,1\n',point);
        tline = fgetl(f1);
    end
end
fclose(f1);
fclose(f2);
    