files = dir(fullfile("wavs/","*.wav"));
disp(files)

for k = 1: length(files)
    baseFileName = files(k).name;
    fullFileName = fullfile("wavs/", baseFileName);
    fprintf('Now reading %s\n', fullFileName)
    [wave, Fs] = audioread(fullFileName);
    stimulus = Stim;
    stimulus.wave = wave;
    stimulus.ID = k;
    save(['stims/',num2str(k),'.mat'], 'stimulus')
end
