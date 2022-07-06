classdef Stim < handle
    %STIM Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        wave
        ID
        info
        calibration_gain
        calibration_target_spl
        calibration_measured_spl
        calibration_mode
        cal_param
        wave_rec
        image
    end
    properties (Constant)
        fs = 195312.50; % realizable sampling rate of RX6 for 200kHz
        fs260k = 2.604166562500000e+05; % realizable sampling rate of RX6 for 260kHz
    end
    
    methods (Static)
        function [wave,info] = do_gen_tone(varargin) % the function that generates tones without modifying object in question
            SAMf = 0; % default use no SAM
            SAMphase = 0; % default is cosine phase
            ramp_dur = 10e-3;
            ramp_type = 'linear';
            freq = nan;
            stim_dur = nan;
            delay = 0;
            verbose = 0;
            max_amp = 0;
            AM = [];
            FS = nan;
            i = 1;
            while i <= length(varargin)
                switch lower(varargin{i})
                    % net_params
                    case 'freq'
                        freq = varargin{i+1};
                    case 'samf' % SAM frequency
                        SAMf = varargin{i+1};
                    case 'samphase'
                        SAMphase = varargin{i+1};
                    case 'ramp_dur' % ramp duration
                        ramp_dur = varargin{i+1};
                    case 'ramp_type' % linear versus cosine gating
                        assert(ismember(varargin{i+1},{'linear','cosine','noramp'}),...
                            'unidentified ramp type!');
                        ramp_type = varargin{i+1};
                    case 'duration' % tone duration
                        stim_dur = varargin{i+1};
                    case 'delay'
                        delay = varargin{i+1};
                    case 'verbose'
                        verbose = varargin{i+1};
                    case 'max_amp'
                        max_amp = varargin{i+1};
                        if max_amp>1
                            warning('maximum amplitude for TDT stim must be below 1!')
                        end
                        %assert(max_amp<=1,'maximum amplitude for stim must be below 1!')
                    case 'am'
                        AM = varargin{i+1};
                        assert(max(abs(AM.envelope))<=1,'envelope amplitude cannot exceed 1!')
                    case 'fs'
                        FS = varargin{i+1};
                    otherwise
                        error('unidentified input param name: %s',varargin{i});
                        
                end
                i = i+2;
            end
            % check if freq is within nyquist range
            assert(freq <= FS/2,...
                sprintf('tone frequency must be below Nyquist frequency (%f)',FS/2));
            assert(~isnan(freq),'frequency of tone required!');
            assert(~isnan(stim_dur),'duration of tone is required!');
            assert(~isnan(FS),'fs is required!');
            % create time axis
            t = 0:1/FS:stim_dur - 1/FS;
            nSampDelay = round(delay * FS);
            nSampSig = length(t);
            
            if verbose
                if ~SAMf
                    fprintf('generating %1.3fsec tone, frequency %5.0fHz, no SAM, %s gating ramp, duration %.1fms\n',...
                        stim_dur,freq,ramp_type,ramp_dur*1000);
                else
                    fprintf('generating %1.3fsec tone, frequency %5.0fHz, SAM freq %2.1fHz, %s gating ramp, duration %.1fms\n',...
                        stim_dur,freq,SAMf,ramp_type,ramp_dur*1000);
                end
            end
            
            % get SAM envelope
            SAMy = Stim.gen_SAM(t,SAMf,SAMphase);
            ramp = Stim.gen_ramp(ramp_dur,ramp_type,nSampSig,FS); % gen ramp
            
            if isempty(AM)
                wave = max_amp * ramp .* SAMy .* sin(2*pi*freq*t + 2*pi*rand); % add a random phase
            else
                % first resample AM to make sure it fits the waveform
                [p,q] = rat(FS / AM.fs_d);
                AM_resamp = resample(AM.envelope,p,q);
                if length(AM_resamp) > nSampSig
                    AM_resamp = AM_resamp(1:nSampSig);
                end
                if length(AM_resamp) < nSampSig
                    AM_resamp = [AM_resamp(:)' zeros(1,nSampSig - length(AM_resamp))];
                end
                wave = max_amp * ramp .* SAMy .* AM_resamp .* sin(2*pi*freq*t + 2*pi*rand);
            end
            % add delay
            wave = [zeros(1,nSampDelay) wave];
            info.freq = freq;
            info.SAMf = SAMf;
            info.SAMphase = SAMphase;
            info.stim_type = 'tone';
            info.nSamp = nSampDelay+nSampSig;
            info.ramp_type = ramp_type;
            info.ramp_dur = ramp_dur;
            info.delay = delay;
            info.nSampDelay = nSampDelay;
            info.stim_dur = stim_dur;
            info.max_amp = max_amp;
            info.AM = AM;
        end
        
        % generating ramp
        function ramp = gen_ramp(varargin)
            % ramp = Stim.gen_ramp(ramp_dur,'linear',nSamp,fs);
            if nargin == 3
                ramp_dur=varargin{1};
                ramp_type=varargin{2};
                nSamp = varargin{3};
                FS = Stim.fs;
            end
            if nargin == 4
                ramp_dur=varargin{1};
                ramp_type=varargin{2};
                nSamp = varargin{3};
                FS = varargin{4};
            end
            % gen_ramp generates a ramp for stimulus
            
            nRamp = floor(ramp_dur*FS);
            ramp = ones(1,nSamp);
            switch ramp_type
                case 'cosine'
                    ramp(1:nRamp) = sin(linspace(0,pi/2,nRamp));
                    ramp(end-nRamp+1:end) = sin(linspace(pi/2,0,nRamp));
                    
                case 'linear'
                    ramp(1:nRamp) = linspace(0,1,nRamp);
                    ramp(end-nRamp+1:end) = linspace(1,0,nRamp);
                case 'noramp'
                    % do nothing
            end
        end
        
        function save_wave_txt(wave,fpath)
            if exist(fpath,'file')
                warning('%s exists, deleting...\n',fpath);
                delete(fpath);
            end
            fid = fopen(fpath,'a+');
            if size(wave,1)~=1
                wave = wave(:)'; % make sure it's row vector
            end
            fprintf(fid,'%f\t',wave);
            fclose(fid);
        end
        
        function save_hit_window_txt(window,fpath)
            if exist(fpath,'file')
                warnin('%s exists, deleting...\n',fpath);
                delete(fpath);
            end
            fid = fopen(fpath,'a+');
            for i = 1:size(window,1)
                fprintf(fid,'%f\t%f',window(i,1),window(i,2));
                if i < size(window,1)
                    fprintf(fid,'\n');
                end
            end
            fclose(fid);
        end
        function save_airpuff_window_txt(window,fpath)
            if exist(fpath,'file')
                warnin('%s exists, deleting...\n',fpath);
                delete(fpath);
            end
            fid = fopen(fpath,'a+');
            for i = 1:size(window,1)
                fprintf(fid,'%f\t%f',window(i,1),window(i,2));
                if i < size(window,1)
                    fprintf(fid,'\n');
                end
            end
            fclose(fid);
        end
        function save_window_txt(window,fpath)
            if exist(fpath,'file')
                warnin('%s exists, deleting...\n',fpath);
                delete(fpath);
            end
            fid = fopen(fpath,'a+');
            for i = 1:size(window,1)
                fprintf(fid,'%f\t%f',window(i,1),window(i,2));
                if i < size(window,1)
                    fprintf(fid,'\n');
                end
            end
            fclose(fid);
        end
        function save_wrong_lick_threshold_txt(nLick,fpath)
            if exist(fpath,'file')
                warnin('%s exists, deleting...\n',fpath);
                delete(fpath);
            end
            fid = fopen(fpath,'a+');
            fprintf(fid,'%d',nLick);
            fclose(fid);
        end
        
        function SAMy = gen_SAM(varargin)
            % gen_SAM(t,SAMf)
            % gen_SAM(t,SAMf,phase)
            % gen_SAM(t,SAMf,phase,modulation_depth)
            if nargin == 2
                t = varargin{1};
                SAMf = varargin{2};
                SAMy = (1 + cos(2*pi*SAMf*t))/2;
            end
            if nargin == 3
                t = varargin{1};
                SAMf = varargin{2};
                phase = varargin{3};
                SAMy = (1 + cos(2*pi*SAMf*t + phase))/2;
            end
        end
        
        function calibrate_speaker()
            % this function generates a mat file that calibrate each
            % frequency to a certain sound level
        end
        
        function [gain,freq,targetSPL] = load_Ji_style_calibration_data(varargin)
            % load_Ed_style_calibration_data load calibration file specified by cali_fn
            % and create a diagonal gain matrix that scales each frequency channel
            % specified by interp_KHz_freqs
            % INPUT:
            % cali_fn          :   file name of calibration file
            % interp_KHz_freqs :   freqs to calibrate, in unit of KHz
            medfilt_ntap = nan;
            if nargin == 1
                cali_fn = varargin{1};
                interp_KHz_freqs = [];
            end
            if nargin == 2
                cali_fn = varargin{1};
                interp_KHz_freqs = varargin{2};
            end
            if nargin == 3
                cali_fn = varargin{1};
                interp_KHz_freqs = varargin{2};
                medfilt_ntap = varargin{3};
            else
                medfilt_ntap = nan;
            end
            % load calibration data
            if ~exist(cali_fn,'file')
                fprintf('error: calibration file does not exist!\n');
            else  % create gain matrix
                fh = load(cali_fn);
                fprintf('using calibration file\n%s\n',cali_fn);
                if ~isempty(interp_KHz_freqs) % when second argument is specified
                    freq = fh.freq;
                    if isnan(medfilt_ntap)
                        dB_gains = 20*log10(fh.A);
                    else
                        dB_gains = 20*log10(medfilt1(fh.A,medfilt_ntap));
                    end
                    
                    interp_dB = interp1(freq,dB_gains,...
                        interp_KHz_freqs,'linear',0);
                    % print a warning if extrapolation was required
                    if (min(interp_KHz_freqs) - min(freq) < -0.01) || ...
                            (max(interp_KHz_freqs) - max(freq) > 0.01)
                        warning('Frequency range outside of calibration range (extrapolate with 0dB)');
                    end
                    if any(interp_dB > 0)
                        warning('Non-negative attenuation in calibration file''%s''',calibfn);
                        %           use_calibration = false;
                    else
                        if any(interp_dB < -48)
                            % attenuating by more than 8 bits can reduce the representation
                            % of the signal below the 24 bit dynamic range of the TDT and
                            % is thus not recommended for software attenuation.
                            warning('Some software attenuations below -48dB, not recommended');
                        end
                    end
                    gain = 10.^(interp_dB/20);
                    
                else % if only one input argument
                    % return calibration file content
                    freq = fh.freq;
                    gain = fh.A;
                end
                targetSPL = fh.target_SPL;
                
            end
        end
        
        function snd_intensity = get_snd_intensity(wave,microphone_sensitivity,preamp_gain,freqrange,fs)
            % ref, absolute hearing threshold
            ref_pascal_RMS = 20 * 1e-6;
            
            wave = wave(:);
            % convert voltage to Pascal
            wave = wave / microphone_sensitivity;
            % correct for preamp_gain
            wave = wave / preamp_gain;
            
            if isnan(freqrange) % if no frequency range is specified
                % calculate mean square as intensity estimate
                measured_pascal_RMS = rms(wave);
                
                
            else
                measured_pascal_RMS = bandpower(wave,fs,freqrange)^0.5;
            end
            
            % convert to dB
            snd_intensity = 20 * log10(measured_pascal_RMS / ref_pascal_RMS);
        end
        
        function [wave_playout,wave_rec,fs] = get_speaker_response(varargin)
            % write wave into RX6 buffer for playout, no additional scaling
            % of wave
            % wave_rec: recorded waveform from RX6
            % wave_playout: user input, should be exactly same as wave
            % fs: sampling rate for sanity check
            
            if nargin == 2
                wav = varargin{1};
                n_fs = varargin{2};
                atten = nan;
                verbose = 1;
            end
            if nargin == 3
                wav = varargin{1};
                n_fs = varargin{2};
                atten = varargin{3};
                verbose = 1;
            end
            if nargin == 4
                wav = varargin{1};
                n_fs = varargin{2};
                atten = varargin{3};
                verbose = varargin{4};
            end
            h = figure('Position',[0,0,1,1]);
            RX6 = actxcontrol('RPco.x', [5 5 0 0],h);
            if verbose
                disp( 'Initializing RX6...');
            end
            if ~RX6.ConnectRX6('GB', 1)
                if verbose
                    disp( 'FAILED connecting to RX6 through GB!');
                    disp( 'Try USB port...');
                end
                if ~RX6.ConnectRX6('USB', 1)
                    error( 'FAILED connecting to RX6 through USB or GB port!');
                else
                    if verbose
                        disp('PASSED connecting to RX6 through USB!');
                    end
                end
            else
                if verbose
                    disp('PASSED connecting to RX6 through GB!');
                end
            end
            
            % connect PA5's
            if verbose
                disp('Connecting to PA5...');
            end
            nPA5 = 1;
            for iRep=1:nPA5
                PA5(iRep) = actxcontrol('PA5.x',[5 5 0 0],h);
                
                % Tell the Active X control to establish comm. link with the PA5s
                %if ~TDTPA5(i).ConnectPA5('USB',i) % xxx - make parameter in handles.TDT
                if ~PA5(iRep).ConnectPA5('GB',iRep)
                    if verbose
                        fprintf('FAILED connecting to PA5 %d/%d through GB!',iRep,nPA5);
                        disp('Try USB port...');
                    end
                    if ~PA5(iRep).ConnectPA5('USB',iRep)
                        error( 'FAILED connecting to PA5 %d/%d through USB or GB port!',iRep,nPA5 );
                    else
                        if verbose
                            fprintf('PASSED connecting to PA5 %d/%d through USB!\n',iRep,nPA5);
                        end
                    end
                else
                    if verbose
                        fprintf('PASSED connecting to PA5 %d/%d through GB!\n',iRep,nPA5);
                    end
                end
                if isnan(atten)
                    PA5(iRep).SetAtten(0);
                else
                    PA5(iRep).SetAtten(atten);
                end
            end
            
            
            % load circuit
            if ~isnan(n_fs)
                if ~RX6.LoadCOFsf('calibration_circuit_v1.0.rcx',n_fs) % the second argument specifies RX6 sampling rate
                    error('FAILED loading circuit!');
                end
            else
                if ~RX6.LoadCOF('calibration_circuit_v1.0.rcx') % the second argument specifies RX6 sampling rate
                    error('FAILED loading circuit!');
                end
            end
            fs = RX6.GetSFreq;
            fprintf('current fs: %f\n',fs);
            
            nSamp = length(wav);
            nSamp2 = round(100e-3 * fs);
            if nSamp < nSamp2
                nAdd = nSamp2 - nSamp;
            else
                nAdd = 0;
            end
            nBlank = round(20e-3*fs)+nAdd;
            nSzCmp = length(wav) + nBlank;
            wav = [wav zeros(1,nBlank)];
            assert(max(abs(wav))<10,'waveform amplitude larger than TDT range!')
            nTotal = size(wav,2);
            
            % write waveform to be played
            if ~RX6.WriteTagV('data',0,wav)
                error('FAILED WriteTagV data!\n');
            end
            
            % set size of buffer to be compared such that play out/recording ends
            if ~RX6.SetTagVal('nSzCmp',nSzCmp)
                error('FAILED SetTagV nSzCmp!\n');
            end
            
            RX6.Run; % start processing chain
            RX6.SoftTrg(1); % kick off circuit
            
            % wait for the waveform to be played out
            while ~RX6.GetTagVal('finished')
                pause(0.2);
                %        RX6.GetTagVal('bufPos2')
            end
            % retrieve waveform recorded
            wave_rec = RX6.ReadTagV('wave_in',0,nTotal);
            if isempty(wave_rec) || length(wave_rec) == 1
                error('FAILED retrieve wave_in!');
            end
            % wave_in = detrend(wave_in,'constant');
            % wave_in = wave_in(1,1:nSamp);
            
            % retrieve waveform written into buffer
            wave_playout = RX6.ReadTagV('wave_out',0,nTotal);
            if isempty(wave_playout) || length(wave_playout) == 1
                error('FAILED retrieve wave_out!');
            end
            % wave_out = wave_out(1,1:nSamp);
            
            RX6.Halt; % stop RX6
            
            close(h);
            if verbose
                fprintf('done\n');
            end
            
        end
        
        function [wave_rec,fs] = get_speaker_response_NI(wave_in,param)
            if ~isfield(param,'fs')
                fs = 200e3;
            else
                fs = param.fs;
            end
            if ~isfield(param,'analogOutChan')
                analogOutChan = 'Dev1\ao0';
            else
                analogOutChan = param.analogOutChan;
            end
            if ~isfield(param,'analogInChan')
                analogInChan = 'Dev1\ai8';
            else
                analogInChan = param.analogInChan;
            end
            e=actxserver('LabVIEW.Application');
            vipath=param.VI_path;
            vi=invoke(e,'GetVIReference',vipath);
            %vi.SetControlValue('analog out channel',analogOutChan);
            %vi.SetControlValue('analog in channel',analogInChan);
            vi.SetControlValue('fs',fs)
            vi.SetControlValue('in',wave_in);
            vi.Run;
            wave_rec = vi.GetControlValue('out');
        end
        
        function out = get_now_str()
            out = datestr(now,'yyyy-mm-dd_HH-MM-SS');
        end
                function fs = get_RX6_fs(n_fs)
            h = figure;
            RX6 = actxcontrol('RPco.x', [5 5 0 0],h);
            disp( 'Initializing RX6...');
            if ~RX6.ConnectRX6('GB', 1)
                disp( 'FAILED connecting to RX6 through GB!');
                disp( 'Try USB port...');
                if ~RX6.ConnectRX6('USB', 1)
                    error( 'FAILED connecting to RX6 through USB or GB port!');
                else
                    disp('PASSED connecting to RX6 through USB!');
                end
            else
                disp('PASSED connecting to RX6 through GB!');
            end
            % load circuit
            if ~isnan(n_fs)
                if ~RX6.LoadCOFsf('calibration_circuit_v1.0.rcx',n_fs) % the second argument specifies RX6 sampling rate
                    error('FAILED loading circuit!');
                end
            else
                if ~RX6.LoadCOF('calibration_circuit_v1.0.rcx') % use default sampling rate specified in circuit
                    error('FAILED loading circuit!');
                end
            end
            fs = RX6.GetSFreq;
            RX6.Halt;
            disp('done');
            close(h);
        end
    end
    
    methods
        function obj = Stim()
            obj.wave = [];
            obj.ID = nan;
            obj.info = [];
            obj.calibration_gain = nan;
            obj.calibration_target_spl = nan;
            obj.calibration_measured_spl = nan;
            obj.calibration_mode = nan;
            obj.image = nan;
        end
        
        function gen_tone(obj,varargin)
            % use static method do_gen_tone
            [obj.wave,obj.info] = Stim.do_gen_tone(varargin{:});
            
        end
        
        function gen_mouse_vocalization(param)
            % param
            % nbout
            % adult vs pup call
            
            
        end
        
        function save_mat(obj,path)
            % path is the folder to save the stim file
            assert(~isnan(obj.ID),'stim ID required for saving!')
            if isfield(obj.info,'stim_type')
                fn = sprintf('stim_%s_%03d.mat',obj.info(1).stim_type,obj.ID);
            else
                fn = sprintf('stim_%03d.mat',obj.ID);
            end
            s = obj; % change of variable name
            if exist(fullfile(path,fn),'file')
                % delete existing file
                warning('deleting existing file...\n');
                delete(fullfile(path,fn));
            end
            save(fullfile(path,fn),'s','-v7.3');
        end
        
        
        
        function gen_dualSAM(obj,varargin)
            % set default
            freq = nan;
            stim_dur = nan;
            SAMf = [0,0]; % default use no SAM
            SAMphase = [0,0]; % default is cosine phase
            ramp_dur = 10e-3;
            ramp_type = 'linear';
            delay = 0;
            
            % parse arguments in varargin
            i = 1;
            while i <= length(varargin)
                switch lower(varargin{i})
                    % net_params
                    case 'freq'
                        % check to see if freq is 2-element vector
                        freq = varargin{i+1};
                        assert(length(freq)==2,'freq should have 2 entries');
                        f1 = freq(1);
                        f2 = freq(2);
                        % check if freq is within nyquist range
                        assert(f1 <= Stim.fs/2 && f2 <= Stim.fs/2,...
                            sprintf('tone frequency must be below Nyquist frequency (%f)',Stim.fs/2));
                    case 'samf' % SAM frequency
                        SAMf = varargin{i+1};
                        assert(length(SAMf)==2,'SAMf should have 2 entries');
                    case 'samphase'
                        SAMphase = varargin{i+1};
                        assert(length(SAMphase)==2,'SAMphase should have 2 entries');
                    case 'ramp_dur' % ramp duration
                        ramp_dur = varargin{i+1};
                    case 'ramp_type' % linear versus cosine gating
                        assert(ismember(varargin{i+1},{'linear','cosine'}),...
                            'unidentified ramp type!');
                        ramp_type = varargin{i+1};
                    case 'duration' % tone duration
                        stim_dur = varargin{i+1};
                    case 'delay'
                        delay = varargin{i+1};
                        assert(length(delay)==2,'delay should have 2 entries');
                    case 'max_amp'
                        max_amp = varargin{i+1};
                        assert(length(max_amp)==2,'max_amp should have 2 entries');
                    otherwise
                        error('unidentified input param name: %s',varargin{i});
                end
                i = i+2;
            end
            
            assert(length(freq)==2,'frequency of tone required!');
            assert(~isnan(stim_dur),'duration of tone is required!');
            
            % generate the first SAM tone
            i = 1;
            [wave1,info1] = Stim.do_gen_tone('freq',freq(i),'duration',stim_dur,...
                'delay',delay(i), 'SAMf',SAMf(i),'SAMphase',SAMphase(i),'ramp_dur',0,... % note we do not apply ramp here
                'ramp_type',ramp_type,'verbose',0,'max_amp',max_amp(i));
            % generate the second SAM tone
            i = 2;
            [wave2,info2] = Stim.do_gen_tone('freq',freq(i),'duration',stim_dur,...
                'delay',delay(i), 'SAMf',SAMf(i),'SAMphase',SAMphase(i),'ramp_dur',0,... % note we do not apply ramp here
                'ramp_type',ramp_type,'verbose',0,'max_amp',max_amp(i));
            % note that wave1 and wave2 do not necessarily have the same
            % length due to different delay
            n1 = length(wave1);
            n2 = length(wave2);
            ntotal = max(n1,n2);
            obj.wave = zeros(1,ntotal);
            obj.wave(1:n1) = obj.wave(1:n1) + wave1; % apply gain on each SAM tone and add them together
            obj.wave(1:n2) = obj.wave(1:n2) + wave2;
            
            % now we add ramp to the waveform
            obj.wave = Stim.gen_ramp(ramp_dur,ramp_type,ntotal) .* obj.wave;
            assert(max(abs(obj.wave))<=1,'waveform out of [-1,1] range!');
            
            dualSAMinfo = cat(1,info1,info2);
            obj.info.dualSAMinfo = dualSAMinfo;
            obj.info.ramp_dur = ramp_dur;
            obj.info.ramp_type = ramp_type;
            obj.info.stim_type = 'dualSAM';
            obj.info.max_amp = max_amp;
            
        end
        
        function [succeeded,measured_SPL,wave_rec,curr_fs] = calibrate_me(obj, cal_param)
            % this function calibrate the mean/peak spl of this stim to be
            % certain level, and stores the information in properties
            ref_pascal_RMS = 20 * 1e-6;
            measured_SPL = -100;
            TDT_range = 10;
            obj.calibration_target_spl = cal_param.target_SPL;
            obj.calibration_mode = cal_param.mode;

            beta = 1;
            n_pass = 1;
            while abs(measured_SPL - cal_param.target_SPL) > cal_param.tol && n_pass <= cal_param.maxIter
                fprintf('Iter%3d\n',n_pass);
                if n_pass == 1
                    if isnan(obj.calibration_gain )
                        obj.calibration_gain = 1;
                    end
                else
                    obj.calibration_gain = beta * 10^((cal_param.target_SPL - measured_SPL)/20) * obj.calibration_gain;
                end
                
                if max(abs(obj.calibration_gain * obj.wave)) > 1 && (~isfield(cal_param,'work_with_NI') || cal_param.work_with_NI == 0)
                    % only worry about this when generating stim for TDT
                    % system
                    warning('hitting the ceiling now\n');
                    obj.calibration_gain =  1/max(abs(obj.wave));
                end
                fprintf('current gain: %f \n',obj.calibration_gain);
                
                if isfield(cal_param,'work_with_NI') && cal_param.work_with_NI == 1
                    wave_tmp = obj.calibration_gain * obj.wave;
                    [wave_rec,curr_fs] = Stim.get_speaker_response_NI(wave_tmp,cal_param);
                else
                    wave_tmp = obj.calibration_gain * obj.wave * TDT_range;
                    [~,wave_rec,curr_fs] = Stim.get_speaker_response(wave_tmp,cal_param.n_fs,...
                    cal_param.atten,cal_param.verbose);
                end
                
                if ~exist('hpFilt','var')
                    hpFilt = designfilt('highpassiir','FilterOrder',10, ...
                        'PassbandFrequency',500,'PassbandRipple',0.2, ...
                        'SampleRate',curr_fs);
                end
                wave_rec = filter(hpFilt,wave_rec);
                %                 wave_in = detrend(wave_in,'constant');
                % break recorded waveform into chunks
                winlength = round(cal_param.window * curr_fs);
                nW = floor(length(wave_rec)/winlength);
                wave_rec = reshape(wave_rec(1:nW*winlength),winlength,nW);
                % for each colomn, calculate RMS
                wave_rec = wave_rec / cal_param.microphone_sensitivity;
                wave_rec = wave_rec / cal_param.preamp_gain;
                % measured_pascal_RMS = rms(wave_in);
                measured_pascal_RMS = bandpower(wave_rec,curr_fs,cal_param.freqband).^0.5;
                % convert to dB
                snd_intensity = 20 * log10(measured_pascal_RMS / ref_pascal_RMS);
                %snd_intensity = Stim.get_snd_intensity(wave_in,cal_param.microphone_sensitivity,cal_param.preamp_gain,...
                %cal_param.freqband,curr_fs);
                switch cal_param.mode
                    case 'peak'
                        measured_SPL = max(snd_intensity);
                    case 'mean'
                        measured_SPL = mean(snd_intensity);
                    case 'median'
                        measured_SPL = median(snd_intensity);
                    case 'window' % look certain window, for example we only want snd level at the start of an FM sweep to be exactly the same for all directions of sweep
                        measured_SPL = mean(snd_intensity(cal_param.windowInd));
                end
                fprintf('measured SPL: %2.1f\n',measured_SPL);
                n_pass = n_pass + 1;
            end
            fprintf('\n')
            if abs(measured_SPL - cal_param.target_SPL) <= cal_param.tol
                succeeded = 1;
            else
                succeeded = 0;
            end
            obj.calibration_measured_spl = measured_SPL;
            if isfield(cal_param,'do_save_rec')
                if cal_param.do_save_rec == 1
                    obj.wave_rec = wave_rec; % note can directly calculate snd level from wave_in, already processed
                end
            else % default is to save
                obj.wave_rec = wave_rec;
            end
            obj.cal_param = cal_param;
        end

    end
end
