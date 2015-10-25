% Matlab startup file.
% Include necessary dependencies
display('Running startup.m');

base = pwd();
fs = filesep();
addpath(pwd);

% folders to be added by genpath
gfolders = {'helper', 'lllvm', 'script', 'thirdparty', 'real_data/ushcn_v2.5/'};
for i=1:length(gfolders)
    fol = gfolders{i};
    p = [base , fs, fol];
    fprintf('add gen path: %s\n', p);
    addpath(genpath(p));
end

addpath('real_data');

clear base fs gfolders i fol p 

