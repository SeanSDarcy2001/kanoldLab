files = dir(fullfile("wavs/","*.wav"));
disp(files)

for k = 1: length(files)
    baseFileName = files(k).name;
    fullFileName = fullfile("wavs/", baseFileName);
    fprintf('Now reading %s\n', fullFileName)
    [wave, Fs] = audioread(fullFileName);
    s = Stim;
    s.wave = wave;
    s.ID = k;
    save(['stims/',num2str(k),'.mat'], 's')
end
