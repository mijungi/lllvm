function [funcs ] = funcs_ushcn( )
%FUNCS_USHCN A collection of functions to deal with weather data in
%real_data/ushcn_v2.5/

    funcs = struct();
    funcs.read_raw_station_strs_tavg = @read_raw_station_strs_tavg;
    funcs.to_numerical = @to_numerical;
    funcs.load_stations_month = @load_stations_month;
    funcs.run_load_stations_month = @run_load_stations_month;
    funcs.plot_stations_2d = @plot_stations_2d;
    funcs.plot_projected_map = @plot_projected_map;
    funcs.load_batch_avg = @load_batch_avg;
    funcs.run_ksn_lwb_cube = @run_ksn_lwb_cube;
    funcs.batch_plot_projections = @batch_plot_projections;
    funcs.subsample_400= @subsample_400;
    funcs.mean_x2d_movie_frames = @mean_x2d_movie_frames;

end

function I=subsample_400()
    % This function is deterministic. 
    % Return a list of length sub of indices which are members of 1:1218.
    %
    oldRng = rng();
    rng(16);
    I = 1:3:1218;
    Isub = randperm(length(I), 400);
    I = I(Isub);
    rng(oldRng);
end

function ksn_lwb = get_ksn_lwb_cube(dataName, ks, seeds, ns)
    % scan all files in saved/ex2 and construct a 3d array of lower bounds
    % of size #k x #seeds x #n 
    % - ks, ns, seeds are array of numbers 
    
    fg = funcs_global();
    %files = fg.expSavedFolder(1);
    ksn_lwb = zeros(length(ks), length(seeds), length(ns));
    for ki=1:length(ks)
        k = ks(ki);
        for si=1:length(seeds)
            s = seeds(si);
            for ni=1:length(ns)
                n = ns(ni);
                fname = sprintf('ushcn-d%s-s%d-k%d-n%d.mat', dataName, s, k, n);
                fpath = fg.expSavedFile(2, fname);
                display(sprintf('loading %s', fpath));
                loaded = load(fpath, 'results');
                % Take the lower bound value in the last iteration.
                ksn_lwb(ki, si, ni) = loaded.results.lwbs(end);
            end
        end
    end
end 

function ksn_lwb = run_ksn_lwb_cube()
    %dataName = 'tmmpr_f05t14_nor';
    %dataName = 'tpravg_f05t14_nor';
    dataName = 'prcp_10y_t14';
    %dataNames = ['tmmpr_f05t14_nor', 'prcp_10y_t14', 'tpravg_f05t14_nor']
    ks = [  2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100, 200 ];
    %seeds = 1:10;
    seeds = 1:3;
    ns = 609;
    ksn_lwb = get_ksn_lwb_cube(dataName, ks, seeds, ns);
    % ks x seeds 
    [max_lwb, maxi] = max(ksn_lwb(:));
    [bki, bsi] = ind2sub(size(ksn_lwb), maxi);

    timeStamp = clock();
    fg = funcs_global();
    fname = sprintf('%s-lwb_cube.mat', dataName);
    savePath = fg.expSavedFile(2, fname);
    save(savePath);

    % export all variables to the base workspace.
    allvars = who;
    warning('off','putvar:overwrite');
    putvar(allvars{:});

end

function batch_plot_projections()
% batch load result files and plot LLLVM 2d projection results.
% Fixed k with different seeds.
%
    
ex = 2;
dataName = 'tmmpr_f05t14_nor';
k = 6;
n = 609;
seed_start = 131;
%subplot_rows = 10;
%subplot_cols = 5;

subplot_rows = 6;
subplot_cols = 5;
total_plots = subplot_rows*subplot_cols;

%seed = 1;
%k = 7;
lalo_pivot = [48.9775; -122.7928];
lola_pivot = [lalo_pivot(2); lalo_pivot(1)];
bottom_mid_pivot = [27.14; -98.12];
fg = funcs_global();
seed = seed_start;
fname = sprintf('ushcn-d%s-s%d-k%d-n%d.mat', dataName, seed, k, n);
fpath = fg.expSavedFile(ex, fname);

figure
for i=1:total_plots
    fname = sprintf('ushcn-d%s-s%d-k%d-n%d.mat', dataName, seed, k, n);
    fpath = fg.expSavedFile(ex, fname);

    while ~exist(fpath, 'file')
        seed = seed + 1;
        fname = sprintf('ushcn-d%s-s%d-k%d-n%d.mat', dataName, seed, k, n);
        fpath = fg.expSavedFile(ex, fname);
    end

    %subplot(subplot_rows, subplot_cols, i);
    subaxis(subplot_rows, subplot_cols, i, 'Spacing', 0.03, 'Padding', 0, 'Margin', 0.05);

    display(sprintf('loading result: %s', fpath));
    % experimental results
    exr = load(fpath, 'subSampleInd', 'data', 'results');
    subSampleInd = exr.subSampleInd;

    lola = [exr.data.laloel(2, subSampleInd); exr.data.laloel(1, subSampleInd)];
    % distances to the pivot point
    dist2pivot = sqrt(sum(bsxfun(@minus, lola, lola_pivot).^2, 1));
    S = sqrt(sum(bsxfun(@minus, lola(2, :), bottom_mid_pivot(2, :)).^2, 1));
    % sizes of the points.
    dist2sizePivot = 1.08.^S/2000 ;
    %dist2sizePivot = 1.1.^S/3000 ;

    % make 2xn 
    mu_x = reshape(exr.results.mean_x, 2, []);
    % plot LL-LVM's projected points
    hold on;
    scatter(mu_x(1, :), mu_x(2, :), dist2sizePivot, dist2pivot, 'fill');
    %set(gca, 'FontSize', 16);
    axis off 
    title(sprintf('s=%d, k=%d, lwb=%.3f', seed, k, exr.results.lwbs(end) ));
    colormap jet;
    %grid on;
    hold off;

    seed = seed + 1;
end


end

function frames = mean_x2d_movie_frames(results, k, subSampleInd, data, movie_dest)
% Generate a sequence of frames showing the time-evolution of mean_x during 
% EM. 
%  - results = a struct returned by LLLVM
%  - data = data struct of the weather data. It has the formdata = 
%         desc: '10-year average monthly precipitation from 2005 to 2014.'
%       laloel: [3x1218 double]
%      station: {1x1218 cell}
%          All: [12x1218x10 double]
%            Y: [12x1218 double]
%          Std: [12x1218 double]
%    timeStamp: [2015 5 28 18 54 38.9886]
%  - movie_dest = a file name to save the movie
%%

lalo_pivot = [48.9775; -122.7928];
bottom_mid_pivot = [27.14; -98.12];
lola_pivot = [lalo_pivot(2); lalo_pivot(1)];

% distances to the pivot point
lola = [data.laloel(2, subSampleInd); data.laloel(1, subSampleInd)];
dist2pivot = sqrt(sum(bsxfun(@minus, lola, lola_pivot).^2, 1));
% sizes of the points.
S = sqrt(sum(bsxfun(@minus, lola(2, :), bottom_mid_pivot(2, :)).^2, 1));
dist2sizePivot = 1.1.^S/3000 ;

em_iters = size(results.mean_x, 2);
%em_iters = 3;
n = size(results.mean_x, 1)/2;
figure
pause(2);
for t=1:em_iters
    mu_x = reshape(results.mean_x(:, t), 2, []);
    % plot LL-LVM's projected points
    scatter(mu_x(1, :), mu_x(2, :), dist2sizePivot, dist2pivot, 'fill');
    set(gca, 'FontSize', 16);
    axis off 
    title(sprintf('LLLVM. k=%d. n=%d. EM iter=%d. lwb=%.5f.', k, n, t, results.lwbs(t) ));

    frames(t) = getframe(gcf);
end

% show the movie 
% use 1st frame to get dimensions
%[h, w, p] = size(frames(1).cdata);
%hf = figure; 
%% resize figure based on frame's w x h, and place at (150, 150)
%set(hf,'Position', [150 150 w h]);
%axis off
% Place frames at bottom left. Play the movie
%movie(hf,frames,4,15,[0 0 0 0]);
%
v = VideoWriter(movie_dest, 'Archival');
v.FrameRate = 2;
open(v);
writeVideo(v, frames)
end

function plot_projected_map(mu_x, k, subSampleInd, data, gplvm_proj)
% 
% Plot the projected stations into 2d along side with the original station positions 
% using the real latitude/longitude.
%
% Input:
%  - ex_results is a struct containing loaded variables from running ex2_ushcn.m.
%

%Y = ex.Y;
%Y = whiten(Y')';
%mu_x = reshape(ex.results.mean_x, 2, []);
G_mux = makeKnnG(mu_x, k);
n = size(mu_x, 2);

lalo_pivot = [48.9775; -122.7928];
bottom_mid_pivot = [27.14; -98.12];
% center
%lalo_pivot = [40.; -95];
% bottom right
%lalo_pivot = [24.55; -81.75];
lola_pivot = [lalo_pivot(2); lalo_pivot(1)];
% distances to the pivot point
subSampleInd = subSampleInd;
lola = [data.laloel(2, subSampleInd); data.laloel(1, subSampleInd)];
dist2pivot = sqrt(sum(bsxfun(@minus, lola, lola_pivot).^2, 1));
% sizes of the points.
S = sqrt(sum(bsxfun(@minus, lola(2, :), bottom_mid_pivot(2, :)).^2, 1));
dist2sizePivot = 1.09.^S/3e3 ;

% plot the true station locations 
fig = plot_stations_2d(data, subSampleInd, lalo_pivot, dist2sizePivot);

% plot LL-LVM's projected points
figure;
hold on;
scatter(mu_x(1, :), mu_x(2, :), dist2sizePivot, dist2pivot, 'fill');
set(gca, 'FontSize', 16);
title(sprintf('LLLVM. k=%d. %d weather stations', k, n));
colormap jet;
%grid on;
hold off;

% plot gplvm results
%gplvm_proj = gplvm(ex.Y', 2)';
if nargin >=5 
    gplvm_proj = gplvm_proj;
    figure;
    hold on;
    scatter(gplvm_proj(1, :), gplvm_proj(2, :), dist2sizePivot, dist2pivot, 'fill');
    set(gca, 'FontSize', 16);
    title(sprintf('GPLVM. %d weather stations', n));
    colormap jet;
    %grid on;
    hold off;

    % plot the graph  for gplvm
    figure 
    set(gca, 'FontSize', 16);
    G_gplvm = makeKnnG(gplvm_proj, k);
    gplot(G_gplvm, lola', '*-');
    title(sprintf('%d-NN graph estimated from GPLVM projections. ', k));
end
%

% plot the graph  for LLLVM
figure 
set(gca, 'FontSize', 16);
gplot(G_mux, lola', '*-');
title(sprintf('LLLVM''s graph. k=%d', k));

end

function fig=plot_stations_2d(sta_data, subSampleInd, lalo_pivot, dist2sizePivot)
%sta_data = 
%
%      station: {1x1218 cell}
%       laloel: [3x1218 double]
%            Y: [12x1218 double]
%    timeStamp: [2015 5 26 21 56 53.0395]
%
% - subSampleInd: list of indices of stations.
% - lalo_pivot is a 2d vector specifying the latitude and longitude of the pivot 
%  point for coloring purpose. The colors of all stations will be based on the 
%  distance to the pivot.     
%  If omitted use the location of 
%  USH00450729,48.9775,-122.7928,18.3,WA, BLAINE       
%  Blaine is a city in Whatcom County, Washington, United States. The city's
%  northern boundary is the Canadian border. This location is at the top-left most 
%  corner.
    n = length(sta_data.station);
    if nargin < 4
        dist2sizePivot = 30;
    end
    if nargin < 3
        lalo_pivot = [48.9775; -122.7928];
    end 
    if nargin < 2
        subSampleInd = 1:n;
    end
    nsub = length(subSampleInd);
    assert(length(lalo_pivot)==2);
    lola_pivot = [lalo_pivot(2); lalo_pivot(1)];
    fig=figure;
    hold on;
    lola = [sta_data.laloel(2, subSampleInd); sta_data.laloel(1, subSampleInd)];
    dist2pivot = sqrt(sum(bsxfun(@minus, lola, lola_pivot(:)).^2, 1));
    
    %scatter(lola(1, :), lola(2, :), 30, dist2pivot, 'fill');
    scatter(lola(1, :), lola(2, :), dist2sizePivot, dist2pivot, 'fill');
    set(gca, 'FontSize', 34);
    title(sprintf('%d weather stations', nsub));
    xlabel('Longitude');
    ylabel('Latitude');
    colormap jet;
    %grid on;
    hold off;
end

function load_batch_avg(prefix, years, fname, desc)
% load all the files starting with prefix_year where year is in years and 
% average the data matrix Y. Write the result to fname.mat in the same folder.
% - desc is a string to be saved for description.
%
    if nargin < 4
        desc = '';
    end
    assert(isnumeric(years));
    fg = funcs_global();
    real_data_folder = fg.realDataFolder();
    dataset_folder = 'ushcn_v2.5';

    laloel = [];
    station = [];
    Yall = {};

    for y=years(:)'
        fn = sprintf('%s_%d.mat', prefix, y);
        fpath = fullfile(real_data_folder, dataset_folder, fn);
        display(sprintf('load %s', fpath));
        data = load(fpath);
        % get the data matrix Y: 12 x #stations.
        laloel = data.laloel;
        station = data.station;
        assert(all(~isnan(data.Y(:))), sprintf('nan value in %s', fpath) );
        Yall{end+1} = data.Y;
    end
    All = cat(3, Yall{:});
    Y = mean(All, 3);
    Std = std(All, [], 3);
    % write a new file 
    dest_path = fullfile(real_data_folder, dataset_folder, fname);
    timeStamp = clock();
    save(dest_path, 'laloel', 'station', 'Y', 'Std', 'All', 'timeStamp', 'desc');
end

function [station, laloel, Y] = run_load_stations_month(year)
    % avg temperature folder
    %data_folder = 'tavg_FLs';
    %fname_tag = 'tavg';
    %raw_stations_reader = @read_raw_station_strs_tavg;
    
    %data_folder = 'prcp_FLs';
    %fname_tag = 'prcp';
    %data_folder = 'tmax_FLs';
    %fname_tag = 'tmax';
    data_folder = 'tmin_FLs';
    fname_tag = 'tmin';
    raw_stations_reader = @read_raw_station_strs;
    %year = 2012;
    % ---------------

    fg = funcs_global();
    real_data_folder = fg.realDataFolder();
    dataset_folder = 'ushcn_v2.5';
    data_folder = fullfile(real_data_folder, dataset_folder, data_folder);
    sta_info = load('stations_info.mat');
    [station, laloel, Y] = load_stations_month(data_folder, raw_stations_reader, sta_info, year);

    % save 
    saveName = sprintf('%s_%d.mat', fname_tag, year);
    savePath = fullfile(real_data_folder, dataset_folder, saveName);
    timeStamp = clock();
    save(savePath, 'station', 'laloel', 'Y', 'timeStamp');
end

function [station, laloel, Y] = load_stations_month(data_folder, raw_stations_reader, sta_info, year)
% Go through all the stations in the sta_info (station info) struct, 
% for each station, load its corresponding data fil in data_folder/ and 
% get the monthly values (e.g., temperatures) for the specified year.
%
% Input:
%  - raw_stations_reader: a function handle. One of {read_raw_station_strs_tavg,
%  read_raw_station_strs}
%
%
% Output:
%  - station: a cell array of station ids.
%  - laloel: 3x#stations to indicate the latitude, longitude, elevation of the
%  stations.
%  - Y: 12 x #stations (12 for 12 months in the specified year). Y may contain 
%  NaN if no data is available for the station in month/year.
%  15 for [latitude, longitude, elevation, month1,  ..., month12]
%
%>> sta_info = load('real_data/ushcn_v2.5/stations_info.mat')
%
%sta_info = 
%
%      station: {1218x1 cell}
%     latitude: [1218x1 double]
%    longitude: [1218x1 double]
%    elevation: [1218x1 double]
%        state: {1218x1 cell}
%         name: {1218x1 cell}
%
    assert(isstruct(sta_info));
    nsta = length(sta_info.station);
    %nsta = min(20, nsta);
    sta_months = zeros(nsta, 12);
    for i=1:nsta
        sta_id = sta_info.station{i};
        % file that begins with the station id. 
        F = dir(fullfile(data_folder, sprintf('%s*', sta_id) ));
        assert(length(F)==1);
        fpath = fullfile(data_folder, F.name);
        display(sprintf('%d: loading station data: %s', i, fpath));
        raw_values = raw_stations_reader(fpath);

        %T: #years x 12
        T = to_numerical(raw_values);
        % filter the year 
        year_row = find(T(:, 1) == year, 1);
        assert(~isempty(year_row), 'specified year does not exist');
        sta_monthly = T(year_row, :);
        sta_months(i, :) = sta_monthly(2:end);
    end

    station = sta_info.station';
    laloel = [sta_info.latitude, sta_info.longitude, sta_info.elevation]';
    Y = sta_months';
end

function T = to_numerical(raw_ym)
% Given a string-valued year-month table loaded from read_raw_station_strs_tavg,
% convert all the values to numerical by removing all the flags. Replace -9999 with nan.
% Output:
%  - T: #years x 12 (for 12 months)
%

    assert(iscell(raw_ym));
    [n, d] = size(raw_ym);
    assert(d==13);
    for r=1:n
        % first column is year 
        for c=2:d
            block = raw_ym{r, c};
            nolast3 = block(1:(end-3));
            raw_ym{r, c} = str2double(nolast3);
        end
    end
    T = cell2mat(raw_ym);
    T(abs(T- -9999) <= 1e-4) = nan;
end

function T = read_raw_station_strs(station_fpath)
% Read from a file where each line is 
%  station year jan_value feb_value mar_value ..... for precipitation values, tmin, tmax.
%
% Expect that the station in each line is the same.
% Return a cell array table of size #years x 13 where the first column is the year, then
% jan_value, .... 
% - year column is numerical. 
% - All month values are string (possibly include flags to indicate how data
% were collected)
%
%

% Auto-generated by MATLAB on 2015/05/28 17:18:50

%% Format string for each line of text:
%   column1: text (%s)
%	column2: double (%f)
%   column3: text (%s)
%	column4: text (%s)
%   column5: text (%s)
%	column6: text (%s)
%   column7: text (%s)
%	column8: text (%s)
%   column9: text (%s)
%	column10: text (%s)
%   column11: text (%s)
%	column12: text (%s)
%   column13: text (%s)
%	column14: text (%s)
% For more information, see the TEXTSCAN documentation.
formatSpec = '%12s%4f%9s%9s%9s%9s%9s%9s%9s%9s%9s%9s%9s%s%[^\n\r]';

%% Open the text file.
fileID = fopen(station_fpath,'r');

%% Read columns of data according to format string.
% This call is based on the structure of the file used to generate this
% code. If an error occurs for a different file, try regenerating the code
% from the Import Tool.
dataArray = textscan(fileID, formatSpec, 'Delimiter', '', 'WhiteSpace', '',  'ReturnOnError', false);

%% Close the text file.
fclose(fileID);

%% Post processing for unimportable data.
% No unimportable data rules were applied during the import, so no post
% processing code is included. To generate code which works for
% unimportable data, select unimportable cells in a file and regenerate the
% script.

%% Allocate imported array to column variable names
station = dataArray{:, 1};
year = dataArray{:, 2};
m1 = dataArray{:, 3};
m2 = dataArray{:, 4};
m3 = dataArray{:, 5};
m4 = dataArray{:, 6};
m5 = dataArray{:, 7};
m6 = dataArray{:, 8};
m7 = dataArray{:, 9};
m8 = dataArray{:, 10};
m9 = dataArray{:, 11};
m10 = dataArray{:, 12};
m11 = dataArray{:, 13};
m12 = dataArray{:, 14};

%% Clear temporary variables
clearvars filename formatSpec fileID dataArray ans;
    T = [num2cell(year), m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12];
end

function T = read_raw_station_strs_tavg(station_fpath)
% Read from a file where each line is 
%  station year jan_value feb_value mar_value ..... for temperature values.
%
% Expect that the station in each line is the same.
% Return a cell array table of size #years x 13 where the first column is the year, then
% jan_value, .... 
% - year column is numerical. 
% - All month values are string (possibly include flags to indicate how data
% were collected)
%
    % Auto-generated by MATLAB on 2015/05/26 20:11:01

    %% Initialize variables.
    filename = station_fpath;

    %% Read columns of data as strings:
    % For more information, see the TEXTSCAN documentation.
    formatSpec = '%11s%6s%8s%9s%9s%9s%9s%9s%9s%9s%9s%9s%9s%9s%[^\n\r]';

    %% Open the text file.
    fileID = fopen(filename,'r');

    %% Read columns of data according to format string.
    % This call is based on the structure of the file used to generate this
    % code. If an error occurs for a different file, try regenerating the code
    % from the Import Tool.
    dataArray = textscan(fileID, formatSpec, 'Delimiter', '', 'WhiteSpace', '',  'ReturnOnError', false);

    %% Close the text file.
    fclose(fileID);

    %% Convert the contents of columns containing numeric strings to numbers.
    % Replace non-numeric strings with NaN.
    raw = repmat({''},length(dataArray{1}),length(dataArray)-1);
    for col=1:length(dataArray)-1
        raw(1:length(dataArray{col}),col) = dataArray{col};
    end
    numericData = NaN(size(dataArray{1},1),size(dataArray,2));

    % Converts strings in the input cell array to numbers. Replaced non-numeric
    % strings with NaN.
    rawData = dataArray{2};
    for row=1:size(rawData, 1);
        % Create a regular expression to detect and remove non-numeric prefixes and
        % suffixes.
        regexstr = '(?<prefix>.*?)(?<numbers>([-]*(\d+[\,]*)+[\.]{0,1}\d*[eEdD]{0,1}[-+]*\d*[i]{0,1})|([-]*(\d+[\,]*)*[\.]{1,1}\d+[eEdD]{0,1}[-+]*\d*[i]{0,1}))(?<suffix>.*)';
        try
            result = regexp(rawData{row}, regexstr, 'names');
            numbers = result.numbers;

            % Detected commas in non-thousand locations.
            invalidThousandsSeparator = false;
            if any(numbers==',');
                thousandsRegExp = '^\d+?(\,\d{3})*\.{0,1}\d*$';
                if isempty(regexp(thousandsRegExp, ',', 'once'));
                    numbers = NaN;
                    invalidThousandsSeparator = true;
                end
            end
            % Convert numeric strings to numbers.
            if ~invalidThousandsSeparator;
                numbers = textscan(strrep(numbers, ',', ''), '%f');
                numericData(row, 2) = numbers{1};
                raw{row, 2} = numbers{1};
            end
        catch me
        end
    end

    %% Split data into numeric and cell columns.
    rawNumericColumns = raw(:, 2);
    rawCellColumns = raw(:, [1,3,4,5,6,7,8,9,10,11,12,13,14]);


    %% Allocate imported array to column variable names
    station = rawCellColumns(:, 1);
    year = cell2mat(rawNumericColumns(:, 1));
    m1 = rawCellColumns(:, 2);
    m2 = rawCellColumns(:, 3);
    m3 = rawCellColumns(:, 4);
    m4 = rawCellColumns(:, 5);
    m5 = rawCellColumns(:, 6);
    m6 = rawCellColumns(:, 7);
    m7 = rawCellColumns(:, 8);
    m8 = rawCellColumns(:, 9);
    m9 = rawCellColumns(:, 10);
    m10 = rawCellColumns(:, 11);
    m11 = rawCellColumns(:, 12);
    m12 = rawCellColumns(:, 13);

    %% Clear temporary variables
    clearvars filename formatSpec fileID dataArray ans raw col numericData rawData row regexstr result numbers invalidThousandsSeparator thousandsRegExp me rawNumericColumns rawCellColumns;

    T = [num2cell(year), m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12];
end
