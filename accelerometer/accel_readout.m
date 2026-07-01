
function accel_readout()
    % Main function wrapper for MATLAB compliance
    
    clc; clear; close all;
    
    %% MATLAB-COMPLIANT CONFIGURATION
    CONFIG = struct();
    CONFIG.channels = [17, 18, 19, 20];
    CONFIG.channel_names = {'Mirror_Y', 'Mirror_X', 'Mirror_Z', 'Desk_Y'};
    CONFIG.sample_rate = 200;
    CONFIG.hardware_gain = 100; % signal conditions gain set to 100
    CONFIG.sensitivity = 0.1; % 100mV/g
    
    CONFIG.log_directory = 'C:\Users\jad507\OneDrive - The Pennsylvania State University\Pictures\Reverse Telescope Test\accel';
    CONFIG.samples_per_file = 300000;  % 5 minutes per file
    CONFIG.display_rate = 4;
    CONFIG.display_seconds = 10;       % Exactly 10 seconds display window
    CONFIG.decimation_factor = 20;
    CONFIG.smoothing_factor = 0.0;
    
    fprintf('=== Reverse Telescope Accelerometer DAQ ===\n');
    
    %% FILE SYSTEM SETUP
    if ~exist(CONFIG.log_directory, 'dir')
        mkdir(CONFIG.log_directory);
    end
    
    session_id = datestr(now, 'yyyy-mm-dd_HHMMSS');
    session_dir = fullfile(CONFIG.log_directory, ['Session_', session_id]);
    mkdir(session_dir);
    fprintf('Session directory: %s\n', session_dir);
    
    %% MATLAB DAQ INITIALIZATION
    try
        dq = daq.createSession('ni');
        for i = 1:length(CONFIG.channels)
            addAnalogInputChannel(dq, 'Dev1', CONFIG.channels(i), 'Voltage');
            dq.Channels(i).Range = [-10, 10];
        end
        dq.Rate = CONFIG.sample_rate;
        dq.IsContinuous = true;
        dq.NotifyWhenDataAvailableExceeds = 200;
        fprintf('DAQ initialized successfully\n');
    catch ME
        error('DAQ failed: %s', ME.message);
    end
    
    %% GLOBAL DATA STRUCTURES (MATLAB style)
    % Use global variables for callback access (MATLAB best practice)
    global DATA_BUFFER DISPLAY_BUFFER SESSION_DATA LOGGER CONFIG_GLOBAL;
    
    CONFIG_GLOBAL = CONFIG;
    
    % Main data buffer (30 minutes)
    BUFFER_SIZE = CONFIG.sample_rate * 60 * 30;
    DATA_BUFFER = struct();
    DATA_BUFFER.timestamps = zeros(BUFFER_SIZE, 1);
    DATA_BUFFER.accel_data = zeros(BUFFER_SIZE, 4);
    DATA_BUFFER.write_index = 1;
    DATA_BUFFER.sample_count = 0;
    DATA_BUFFER.smoothed = zeros(1, 4);
    
    % Display buffer
    DISPLAY_SIZE = CONFIG.sample_rate * CONFIG.display_seconds;
    DISPLAY_BUFFER = struct();
    DISPLAY_BUFFER.time = zeros(DISPLAY_SIZE, 1);
    DISPLAY_BUFFER.data = zeros(DISPLAY_SIZE, 4);
    DISPLAY_BUFFER.write_index = 1;
    DISPLAY_BUFFER.sample_count = 0;
    DISPLAY_BUFFER.decimation_counter = 0;
    
    % Session tracking
    SESSION_DATA = struct();
    SESSION_DATA.start_time = now;
    SESSION_DATA.start_tic = tic;
    SESSION_DATA.last_status = now;
    SESSION_DATA.stop_requested = false;
    SESSION_DATA.session_dir = session_dir;
    SESSION_DATA.session_id = session_id;
    
    % Logger structure
    LOGGER = struct();
    LOGGER.file_counter = 1;
    LOGGER.current_file = '';
    LOGGER.samples_in_file = 0;
    LOGGER.total_samples = 0;
    LOGGER.fid = -1;  % File handle for binary writing
    
    fprintf('Pre-allocated %.0f MB for data buffers\n', (BUFFER_SIZE * 5 * 8) / 1e6);
    
    %% CLEAN MATLAB GRAPHICS SETUP
    fig = figure('Name', 'Reverse Telescope Accelerometer DAQ', ...
        'Position', [50, 50, 1400, 900], ...
        'Color', 'white');
    
    % Store figure handle globally
    SESSION_DATA.figure_handle = fig;
    
    % Create subplots with MATLAB-compliant colors
    colors = {[0, 0.4, 0.8], [0.8, 0.3, 0.1], [0.9, 0.7, 0.1], [0.5, 0.2, 0.6]};
    ax_handles = [];
    line_handles = [];
    
    for i = 1:4
        ax_handles(i) = subplot(4, 1, i);
        line_handles(i) = plot(ax_handles(i), 0, 0, 'Color', colors{i}, 'LineWidth', 2);
        
        % REMOVED: Individual subplot titles for cleaner look
        % title(ax_handles(i), sprintf('%s Acceleration', CONFIG.channel_names{i}), ...
        %     'FontSize', 14, 'FontWeight', 'bold');
        
        % Clean y-axis label with channel name
        ylabel(ax_handles(i), sprintf('%s (g)', strrep(CONFIG.channel_names{i}, '_', ' ')), ...
            'FontSize', 12, 'FontWeight', 'bold');
        grid(ax_handles(i), 'on');
        xlim(ax_handles(i), [0, CONFIG.display_seconds]);
        ylim(ax_handles(i), [-1, 1]);
    end
    
    % Clean x-axis label
    xlabel(ax_handles(4), 'Clock Time (HH:MM:SS) - Last 10 Seconds', ...
        'FontSize', 12, 'FontWeight', 'bold');
    
    % REMOVED: Overall title for cleaner appearance
    % sgtitle('BULLETPROOF 12-Hour System: 10-Second Rolling Window', 'FontSize', 16, 'FontWeight', 'bold');
    
    % Status displays
    status_text = uicontrol('Style', 'text', 'Units', 'normalized', ...
        'Position', [0.01, 0.94, 0.98, 0.05], ...
        'String', 'Initializing Display', ...
        'FontSize', 12, 'FontWeight', 'bold', ...
        'BackgroundColor', [0.9, 1, 0.9]);
    
    % Store handles globally
    SESSION_DATA.ax_handles = ax_handles;
    SESSION_DATA.line_handles = line_handles;
    SESSION_DATA.status_text = status_text;
    
    % Initialize first binary file
    create_new_binary_file();
    
    %% DATA CALLBACK - MATLAB STYLE
    data_listener = addlistener(dq, 'DataAvailable', @data_callback_matlab);
    SESSION_DATA.data_listener = data_listener;
    
    %% DISPLAY TIMER - MATLAB COMPLIANT
    display_timer = timer();
    set(display_timer, 'ExecutionMode', 'fixedRate');
    set(display_timer, 'Period', 1/CONFIG.display_rate);
    set(display_timer, 'TimerFcn', @display_callback_time_fixed);  % UPDATED: Time-fixed display
    SESSION_DATA.display_timer = display_timer;
    
    %% KEYBOARD HANDLER - MATLAB STYLE
    set(fig, 'KeyPressFcn', @keyboard_callback_matlab);
    set(fig, 'CloseRequestFcn', @cleanup_callback_matlab);
    
    %% START SYSTEM
    fprintf('\n=== INITIALIZING ===\n');
    
    try
        start(display_timer);
        startBackground(dq);
        
        fprintf('=== RUNNING ===\n');
        fprintf('Press SPACE for manual save, ESC to stop\n\n');

        
        % Main loop - MATLAB style
        while ishandle(fig) && ~SESSION_DATA.stop_requested
            pause(1);
            
            % Update global SESSION_DATA to check stop condition
            global SESSION_DATA;
            if SESSION_DATA.stop_requested
                break;
            end
        end
        
    catch ME
        fprintf('System error: %s\n', ME.message);
    end
    
    %% CLEANUP
    cleanup_system_matlab();
    
end % End of main function

%% MATLAB CALLBACK FUNCTIONS (separate functions, not nested)

function data_callback_matlab(~, event)
    % MATLAB-compliant data processing callback
    global DATA_BUFFER DISPLAY_BUFFER SESSION_DATA LOGGER CONFIG_GLOBAL;
    
    try
        raw_data = event.Data;
        timestamps = event.TimeStamps;
        n_samples = size(raw_data, 1);
        
        for k = 1:n_samples
            % Convert to g-forces
            raw_g = (raw_data(k, :) / CONFIG_GLOBAL.hardware_gain) / CONFIG_GLOBAL.sensitivity;
            DATA_BUFFER.smoothed = CONFIG_GLOBAL.smoothing_factor * DATA_BUFFER.smoothed + ...
                (1 - CONFIG_GLOBAL.smoothing_factor) * raw_g;
            
            % Store in buffer
            idx = DATA_BUFFER.write_index;
            DATA_BUFFER.timestamps(idx) = timestamps(k);
            DATA_BUFFER.accel_data(idx, :) = DATA_BUFFER.smoothed;
            
            % Update pointers
            DATA_BUFFER.write_index = DATA_BUFFER.write_index + 1;
            if DATA_BUFFER.write_index > length(DATA_BUFFER.timestamps)
                DATA_BUFFER.write_index = 1;
            end
            DATA_BUFFER.sample_count = DATA_BUFFER.sample_count + 1;
            
            % Binary logging - MATLAB style
            log_sample_to_binary_matlab(timestamps(k), DATA_BUFFER.smoothed);
            
            % Display buffer update
            DISPLAY_BUFFER.decimation_counter = DISPLAY_BUFFER.decimation_counter + 1;
            if DISPLAY_BUFFER.decimation_counter >= CONFIG_GLOBAL.decimation_factor
                update_display_buffer_matlab(timestamps(k), DATA_BUFFER.smoothed);
                DISPLAY_BUFFER.decimation_counter = 0;
            end
        end
    catch
        % Silent error handling
    end
end

function display_callback_time_fixed(~, ~)
    % FIXED: HH:MM:SS rolling time display callback
    global DISPLAY_BUFFER SESSION_DATA CONFIG_GLOBAL DATA_BUFFER;
    
    try
        if ~ishandle(SESSION_DATA.figure_handle)
            return;
        end
        
        if DISPLAY_BUFFER.sample_count < 50
            return;
        end
        
        % Get display data
        n_samples = min(DISPLAY_BUFFER.sample_count, length(DISPLAY_BUFFER.time));
        
        if DISPLAY_BUFFER.write_index == 1
            indices = 1:n_samples;
        else
            if DISPLAY_BUFFER.sample_count < length(DISPLAY_BUFFER.time)
                indices = 1:DISPLAY_BUFFER.sample_count;
            else
                % MATLAB-compliant array concatenation
                part1 = DISPLAY_BUFFER.write_index:length(DISPLAY_BUFFER.time);
                part2 = 1:(DISPLAY_BUFFER.write_index-1);
                indices = [part1, part2];
            end
        end
        
        time_data = DISPLAY_BUFFER.time(indices);
        accel_data = DISPLAY_BUFFER.data(indices, :);
        
        % FIXED: Use elapsed time from session start for display
        if length(time_data) > 1
            % Calculate elapsed time from session start
            elapsed_time = time_data - time_data(1);
            
            % Find data points within the last 10 seconds
            max_elapsed = elapsed_time(end);
            min_elapsed = max(0, max_elapsed - CONFIG_GLOBAL.display_seconds);
            
            % Filter to only show last 10 seconds
            valid_indices = elapsed_time >= min_elapsed;
            time_display = elapsed_time(valid_indices);
            accel_display = accel_data(valid_indices, :);
            
            % FIXED: Calculate actual clock time for x-axis labels
            session_start_seconds = (SESSION_DATA.start_time - floor(SESSION_DATA.start_time)) * 24 * 3600;
            current_session_time = session_start_seconds + max_elapsed;
            
        else
            time_display = time_data;
            accel_display = accel_data;
            current_session_time = 0;
        end
        
        % Update plots with rolling data
        for ch = 1:4
            set(SESSION_DATA.line_handles(ch), 'XData', time_display, 'YData', accel_display(:, ch));
            
            % FIXED: Set x-axis limits using elapsed time
            if length(time_display) > 1
                xlim(SESSION_DATA.ax_handles(ch), [time_display(1), time_display(end)]);
                
                % FIXED: Create HH:MM:SS tick labels for better readability
                if mod(DATA_BUFFER.sample_count, 2500) == 0  % Update labels occasionally
                    % Calculate time points for ticks (every 2 seconds within the 10-second window)
                    tick_interval = 2;  % seconds
                    n_ticks = floor(CONFIG_GLOBAL.display_seconds / tick_interval) + 1;
                    
                    if length(time_display) > 10
                        tick_positions = linspace(time_display(1), time_display(end), n_ticks);
                        tick_labels = cell(1, length(tick_positions));
                        
                        for t = 1:length(tick_positions)
                            % Calculate absolute time for this tick
                            elapsed_at_tick = tick_positions(t);
                            absolute_seconds = session_start_seconds + elapsed_at_tick + time_data(1);
                            
                            % Convert to HH:MM:SS
                            hours = floor(absolute_seconds / 3600);
                            minutes = floor(mod(absolute_seconds, 3600) / 60);
                            seconds = mod(absolute_seconds, 60);
                            
                            tick_labels{t} = sprintf('%02d:%02d:%02d', hours, minutes, floor(seconds));
                        end
                        
                        set(SESSION_DATA.ax_handles(ch), 'XTick', tick_positions);
                        set(SESSION_DATA.ax_handles(ch), 'XTickLabel', tick_labels);
                        set(SESSION_DATA.ax_handles(ch), 'XTickLabelRotation', 0);
                    end
                end
            else
                xlim(SESSION_DATA.ax_handles(ch), [0, CONFIG_GLOBAL.display_seconds]);
            end
            
            % Auto-scale y-axis
            if mod(DATA_BUFFER.sample_count, 5000) == 0 && length(accel_display) > 20
                y_range = [min(accel_display(:, ch)), max(accel_display(:, ch))];
                if y_range(2) > y_range(1)
                    margin = 0.1 * (y_range(2) - y_range(1));
                    ylim(SESSION_DATA.ax_handles(ch), [y_range(1) - margin, y_range(2) + margin]);
                end
            end
        end
        
        % Status update
        if (now - SESSION_DATA.last_status) * 24 * 3600 >= 5
            update_status_matlab();
            SESSION_DATA.last_status = now;
        end
        
        drawnow limitrate;
        
    catch
        % Silent error
    end
end

function keyboard_callback_matlab(~, event)
    % MATLAB keyboard handling
    global SESSION_DATA LOGGER;
    
    switch event.Key
        case 'space'
            % Manual flush
            if LOGGER.fid ~= -1
                fflush(LOGGER.fid);  % MATLAB function to flush file
            end
            fprintf('Manual flush completed\n');
            
        case 'escape'
            % Stop system
            SESSION_DATA.stop_requested = true;
            set(SESSION_DATA.status_text, 'String', 'STOPPING - Saving data...', ...
                'BackgroundColor', [1, 1, 0.8]);
            drawnow;
    end
end

function cleanup_callback_matlab(~, ~)
    % Handle figure close
    global SESSION_DATA;
    SESSION_DATA.stop_requested = true;
    cleanup_system_matlab();
end

%% MATLAB UTILITY FUNCTIONS

function log_sample_to_binary_matlab(timestamp, accel_data)
    % MATLAB-compliant binary logging
    global LOGGER CONFIG_GLOBAL SESSION_DATA;
    
    try
        % Check if new file needed
        if LOGGER.samples_in_file >= CONFIG_GLOBAL.samples_per_file
            close_current_binary_file_matlab();
            create_new_binary_file();
        end
        
        % Write binary data - MATLAB style
        if LOGGER.fid ~= -1
            sample_data = [timestamp, accel_data];  % 1x5 vector
            count = fwrite(LOGGER.fid, sample_data, 'double');
            
            if count == 5
                LOGGER.samples_in_file = LOGGER.samples_in_file + 1;
                LOGGER.total_samples = LOGGER.total_samples + 1;
            end
        end
        
    catch
        % Silent error
    end
end

function create_new_binary_file()
    % Create new binary file - MATLAB compliant
    global LOGGER SESSION_DATA CONFIG_GLOBAL;
    
    try
        % Create filename
        filename = sprintf('AccelData_%s_File%04d.bin', SESSION_DATA.session_id, LOGGER.file_counter);
        filepath = fullfile(SESSION_DATA.session_dir, filename);
        
        % Open binary file for writing
        LOGGER.fid = fopen(filepath, 'wb');  % 'wb' = write binary
        
        if LOGGER.fid ~= -1
            LOGGER.current_file = filepath;
            LOGGER.samples_in_file = 0;
            fprintf('Created binary file %d\n', LOGGER.file_counter);
        else
            fprintf('Warning: Failed to create file %d\n', LOGGER.file_counter);
        end
        
    catch ME
        fprintf('File creation error: %s\n', ME.message);
    end
end

function close_current_binary_file_matlab()
    % Close current binary file and create CSV companion
    global LOGGER SESSION_DATA;
    
    try
        if LOGGER.fid ~= -1
            fclose(LOGGER.fid);
            LOGGER.fid = -1;
        end
        
        % Create CSV companion file
        if LOGGER.samples_in_file > 0
            create_csv_companion_matlab();
        end
        
        LOGGER.file_counter = LOGGER.file_counter + 1;
        
    catch
        % Silent error
    end
end

function create_csv_companion_matlab()
    % Create CSV version of binary file for analysis
    global LOGGER SESSION_DATA;
    
    try
        % Read binary file back
        bin_fid = fopen(LOGGER.current_file, 'rb');
        if bin_fid == -1, return; end
        
        % Read all data
        data_matrix = fread(bin_fid, [5, LOGGER.samples_in_file], 'double');
        fclose(bin_fid);
        
        if isempty(data_matrix), return; end
        
        % Transpose to get samples as rows
        data_matrix = data_matrix';  % MATLAB transpose
        
        % Create CSV filename
        csv_filename = strrep(LOGGER.current_file, '.bin', '.csv');
        
        % Write CSV file
        csv_fid = fopen(csv_filename, 'w');
        if csv_fid == -1, return; end
        
        % Write header
        fprintf(csv_fid, 'AbsoluteTime,RelativeTime_s,Mirror_Y_g,Mirror_X_g,Mirror_Z_g,Desk_Y_g\n');
        
        % Write data
        for i = 1:size(data_matrix, 1)
            abs_time = SESSION_DATA.start_time + data_matrix(i,1)/(24*3600);
            fprintf(csv_fid, '%s,%.6f,%.6f,%.6f,%.6f,%.6f\n', ...
                datestr(abs_time, 'yyyy-mm-dd HH:MM:SS.FFF'), ...
                data_matrix(i, 1), data_matrix(i, 2), data_matrix(i, 3), ...
                data_matrix(i, 4), data_matrix(i, 5));
        end
        
        fclose(csv_fid);
        
    catch
        % Silent error
    end
end

function update_display_buffer_matlab(timestamp, accel_data)
    % Update display buffer - MATLAB style
    global DISPLAY_BUFFER;
    
    idx = DISPLAY_BUFFER.write_index;
    DISPLAY_BUFFER.time(idx) = timestamp;
    DISPLAY_BUFFER.data(idx, :) = accel_data;
    
    DISPLAY_BUFFER.write_index = DISPLAY_BUFFER.write_index + 1;
    if DISPLAY_BUFFER.write_index > length(DISPLAY_BUFFER.time)
        DISPLAY_BUFFER.write_index = 1;
    end
    
    DISPLAY_BUFFER.sample_count = min(DISPLAY_BUFFER.sample_count + 1, length(DISPLAY_BUFFER.time));
end

function update_status_matlab()
    % Update status display
    global SESSION_DATA DATA_BUFFER LOGGER;
    
    try
        elapsed_hours = (now - SESSION_DATA.start_time) * 24;
        actual_rate = DATA_BUFFER.sample_count / toc(SESSION_DATA.start_tic);
        
        status_str = sprintf('%.2f hrs | %.1fM samples | %.0f Hz | File %d ', ...
            elapsed_hours, DATA_BUFFER.sample_count/1e6, actual_rate, LOGGER.file_counter);
        
        set(SESSION_DATA.status_text, 'String', status_str);
        
    catch
    end
end

function cleanup_system_matlab()
    % Final cleanup - MATLAB style
    global SESSION_DATA DATA_BUFFER LOGGER;
    
    fprintf('\n=== SYSTEM CLEANUP ===\n');
    
    try
        % Stop timer
        if isfield(SESSION_DATA, 'display_timer') && isvalid(SESSION_DATA.display_timer)
            stop(SESSION_DATA.display_timer);
            delete(SESSION_DATA.display_timer);
        end
        
        % Stop DAQ
        if isfield(SESSION_DATA, 'data_listener') && isvalid(SESSION_DATA.data_listener)
            delete(SESSION_DATA.data_listener);
        end
        
        % Close current file
        close_current_binary_file_matlab();
        
        % Statistics
        runtime_hours = (now - SESSION_DATA.start_time) * 24;
        fprintf('Runtime: %.2f hours\n', runtime_hours);
        fprintf('Total samples: %.0f\n', DATA_BUFFER.sample_count);
        fprintf('Files created: %d\n', LOGGER.file_counter);
        fprintf('Session directory: %s\n', SESSION_DATA.session_dir);
        
        % Create summary
        create_session_summary_matlab();
        
    catch ME
        fprintf('Cleanup error: %s\n', ME.message);
    end
    
    % Close figure
    if isfield(SESSION_DATA, 'figure_handle') && ishandle(SESSION_DATA.figure_handle)
        delete(SESSION_DATA.figure_handle);
    end
    
    fprintf('Shutdown complete\n');
end

function create_session_summary_matlab()
    % Create session summary file
    global SESSION_DATA DATA_BUFFER LOGGER CONFIG_GLOBAL;
    
    try
        summary_file = fullfile(SESSION_DATA.session_dir, 'session_summary.txt');
        fid = fopen(summary_file, 'w');
        if fid ~= -1
            fprintf(fid, 'SESSION SUMMARY\n');
            fprintf(fid, '==================================\n\n');
            fprintf(fid, 'Start: %s\n', datestr(SESSION_DATA.start_time));
            fprintf(fid, 'End: %s\n', datestr(now));
            fprintf(fid, 'Duration: %.2f hours\n', (now - SESSION_DATA.start_time) * 24);
            fprintf(fid, 'Samples: %.0f\n', DATA_BUFFER.sample_count);
            fprintf(fid, 'Files: %d\n', LOGGER.file_counter);
            fprintf(fid, 'Channels: %s\n', strjoin(CONFIG_GLOBAL.channel_names, ', '));
            fprintf(fid, 'Display: HH:MM:SS rolling window\n');
            fclose(fid);
        end
    catch
    end
end