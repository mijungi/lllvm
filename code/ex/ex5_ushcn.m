function [ ] = ex5_ushcn( dataName, seed, k )
%EX5_USHCN An experiment on USHCN climate data using lllvm_1ep
% Input:
%  - seed: random seed
%  - k: k in kNN to construct the graph G
%

oldRng = rng();
rng(seed);
display(sprintf('starting %s(%s, %d, %d)', mfilename, dataName, seed, k));

%dataName = 'tavg_2014';
data = load(sprintf('%s.mat', dataName));
%data = 
%      station: {1x1218 cell}
%       laloel: [3x1218 double]
%            Y: [12x1218 double]
%    timeStamp: [2015 5 26 21 56 53.0395]

% half of 1218 stations.
%subSampleInd = 1:2:size(data.Y, 2);

% 400 
funcs = funcs_ushcn();
subSampleInd = funcs.subsample_400();
%
% all stations
%subSampleInd = 1:size(data.Y, 2);
Y = data.Y(:, subSampleInd);
[dy, n] = size(Y);


saveFName = sprintf('ushcn-d%s-s%d-k%d-n%d.mat', dataName, seed, k, n);
fglobal = funcs_global();
fpath = fglobal.expSavedFile(5, saveFName);
rerun = true;
if ~rerun && exist(fpath, 'file')
    % skip the experiment if the result already exists
    return;
end

%% construct the graph
% k = number of neighbours in kNN
G = makeKnnG(Y, k);
dx = 2;

%% options to lllvm. Include initializations
op = struct();
op.seed = seed;
op.max_em_iter = 50;
op.abs_tol = 1e-1;
op.G = G;
op.dx = dx;
op.alpha0 = 1;
op.gamma0 = 1;
op.epsilon = 1e-3;

L = diag(sum(G, 1)) - G;
invOmega = kron(2*L, eye(dx));
J = kron(ones(n,1), eye(dx));
op.cov_c0 = inv(op.epsilon*(J*J') + invOmega) ;
% initial value of mean_c
op.mean_c0 = randn(dy, dx*n)*op.cov_c0';

%recorder = create_recorder('print_struct');
store_every_iter = 10;
only_cov_diag = false;
%recorder = create_recorder_store_latent(store_every_iter, only_cov_diag);
recorder = create_recorder_hyper();
op.recorder = recorder;


%% Run lllvm 
[ results, op ] = lllvm_1ep(Y, op);

% rec_vars will contains all the recorded variables.
rec_vars = recorder();

% gplvm 
gplvm_proj = gplvm(Y', 2)';

%% write all results to a file 
timestamp = clock();

save(fpath, 'timestamp', 'rec_vars', 'results', 'op', 'Y', 'k', ...
    'subSampleInd', 'data', 'gplvm_proj');

% export all variables to the base workspace.
%allvars = who;
%warning('off','putvar:overwrite');
%putvar(allvars{:});

display(op);
display(results);


% change seed back
rng(oldRng);

end

