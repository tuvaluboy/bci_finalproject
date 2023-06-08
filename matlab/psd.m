% Loop through each dataset in ALLEEG
for k = 1:length(ALLEEG)
    
    % Make the k-th dataset the current dataset
    EEG = ALLEEG(k);
    CURRENTSET = k;
    
    % Check if the dataset is empty
    if isempty(EEG.data)
        fprintf('Dataset %d is empty. Skipping...\n', k);
        continue; % Skip to the next iteration
    end
    
    % Define parameters
    channels = 1:14; % Select first 14 channels
    chan_labels = {'Fp1', 'Fp2', 'F3', 'F4', 'T3', 'C3', 'Cz', 'C4', 'T4', 'P3', 'Pz', 'P4', 'O1', 'O2'}; % Your channel labels
    
    % Extract data for selected channels
    data_input = EEG.data(channels, :);
    
    % Create a new EEG structure with only the selected channels
    newEEG = EEG;
    newEEG.data = data_input;
    newEEG.nbchan = length(channels); % Update the number of channels
    
    % Assign channel labels to newEEG.chanlocs
    for i = 1:length(chan_labels)
        newEEG.chanlocs(i).labels = chan_labels{i};
    end
    
    % Power Spectral Density (PSD) for the 14 selected channels
    figure; % create new figure for each dataset
    spectopo(newEEG.data, 0, newEEG.srate, 'percent', 15);
end
