% Start EEGLAB
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;

% Define folder path
folderPath = 'folder location';

% Get a list of all .mat files in the folder
files = dir(fullfile(folderPath, '*.mat'));

% Define number of channels and sampling rate (replace with your actual values)
nb_chan = 62;
srate = 200; % replace with your actual sampling rate

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

    % Update EEGLAB
    [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG);
end

% Redraw EEGLAB
eeglab redraw;
