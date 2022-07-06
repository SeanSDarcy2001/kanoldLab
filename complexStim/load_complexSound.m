info = audioinfo('sound_files/rufus.m4a');
[y,Fs] = audioread('sound_files/rufus.m4a');
disp(Fs)
y = resample(y, 6, 1);
Fs = 260417;
t = 0:seconds(1/Fs):seconds(info.Duration);
%t = t(1:end-1);
%t = t(Fs:2*Fs);
%plot(t,y(Fs:2*Fs))
xlabel('Time')
ylabel('Audio Signal')

disp(info.TotalSamples/Fs)
subplot
for i = 1 : info.Duration - 1
    subplot(75, 4, i)
    if i == 1 
        sample = t(1:Fs);
        sig = y(1:Fs);
        wavFilename = "./wavs/" + string(i) + '.wav' ;
        audiowrite(wavFilename,sig,Fs);
    else
        sample = t(i*Fs:(i+1)*Fs);
        sig = y(i*Fs:(i+1)*Fs);
        wavFilename = "./wavs/" + string(i) + '.wav';
        audiowrite(wavFilename,sig,Fs);
    end
    plot(sample,sig)
    xlabel('Time')
    ylabel('Audio Signal')
end