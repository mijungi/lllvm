function fu = funcs_global()
    %FUNCS_GLOBAL Return a struct containing functions related to the global environment 
    %of the project.
    fu = struct();
    fu.getRootFolder = @getRootFolder;
    fu.getScriptFolder = @getScriptFolder;
    fu.expSavedFolder = @expSavedFolder;
    fu.expSavedFile = @expSavedFile;
    fu.scriptSavedFile = @scriptSavedFile;
    fu.scriptSavedFolder = @scriptSavedFolder;
    fu.realDataFolder = @realDataFolder;
end

function r=getRootFolder()
    [p,f,e]=fileparts(which('startup.m'));
    r=p;
end

function f=getScriptFolder()
    % return the folder containing directly runnable scripts
    p=getRootFolder();
    f=fullfile(p, 'script');
    if ~exist(f, 'dir')
        mkdir(f);
    end
end

function f=getSavedFolder()
    % return the top folder for saving .mat files 

    p=getRootFolder();
    f=fullfile(p, 'saved');
    if ~exist(f, 'dir')
        mkdir(f);
    end
end


function fpath=expSavedFile(expNum, fname)
    % construct a full path the the file name fname in the experiment 
    % folder identified by expNum
    expNumFolder=expSavedFolder(expNum);
    fpath=fullfile(expNumFolder, fname);
end

function expNumFolder=expSavedFolder(expNum)
    % return full path to folder used for saving results of experiment 
    % identified by expNum. Create the folder if not existed. 
    assert(isscalar(expNum));
    assert(mod(expNum, 1)==0);
    root=getSavedFolder();

    fname=sprintf('ex%d', expNum);
    expNumFolder=fullfile(root, fname);
    if ~exist(expNumFolder, 'dir')
        mkdir(expNumFolder);
    end
end

function fpath=scriptSavedFile(fname)
   savedFolder=scriptSavedFolder();
   fpath=fullfile(savedFolder, fname);
end

function savedScript=scriptSavedFolder()
   % return full path to folder used for saving results of temporary scripts 
   saved=getSavedFolder();
   savedScript=fullfile(saved, 'script');
   if ~exist(savedScript, 'dir')
       mkdir(savedScript);
   end

end

function realFolder = realDataFolder()
    r = getRootFolder();
    realFolder = fullfile(r, 'real_data');

end
