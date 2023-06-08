% Start EEGLAB
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;

% Define folder path
folderPath = '';

% Get a list of all .mat files in the folder
files = dir(fullfile(folderPath, '*.mat'));

% Define number of channels and sampling rate (replace with your actual values)
nb_chan = 62;
srate = 200; % replace with your actual sampling rate

% Channel labels
chan_labels = {'Fp1', 'Fp2', 'F3', 'F4', 'T3', 'C3', 'Cz', 'C4', 'T4', 'P3', 'Pz', 'P4', 'O1', 'O2'};

% Loop through each file in the folder
for k = 1:length(files)
    % Get the full file name
    fullFileName = fullfile(files(k).folder, files(k).name);

    % Load .mat file
    loaded_data = load(fullFileName);

    % Get the variable names in the loaded file
    variableNames = fieldnames(loaded_data);

    % Extract the EEG data from the first variable in the file
    EEG_data = loaded_data.(variableNames{1});

    % Import data into EEGLAB
    EEG = pop_importdata('dataformat', 'array', 'nbchan', nb_chan, 'data', 'EEG_data', 'srate', srate, 'pnts', 0, 'xmin', 0);

    % Keep only the first 14 channels
    EEG = pop_select(EEG, 'channel', 1:14);

    % Create a new chanlocs structure with these labels
    for i = 1:length(chan_labels)
        EEG.chanlocs(i).labels = chan_labels{i};
    end

    % Set the standard 10-20 locations
    EEG = pop_chanedit(EEG, 'lookup','standard-10-5-cap385.elp');

    % Update EEGLAB
    [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG);
end

% Redraw EEGLAB
eeglab redraw;
