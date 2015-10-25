function [ ] = test_ushcn()
%TEST_USHCN Run LL-LVM on USHCN climate data.
%
%@author Wittawat. Created 27 May 2015.
%

seed = 4;
oldRng = rng();
rng(seed);

%dataName = 'tavg_2014';
%dataName = 'tmmpr_f05t14_nor';
dataName = 'tpravg_f05t14_nor';
data = load(sprintf('%s.mat', dataName));
%data = 
%      station: {1x1218 cell}
%       laloel: [3x1218 double]
%            Y: [12x1218 double]
%    timeStamp: [2015 5 26 21 56 53.0395]

% half of 1218 stations.
subSampleInd = 1:2:size(data.Y, 2);
Y = data.Y(:, subSampleInd);
[dy, n] = size(Y);

%% construct the graph
% k = number of neighbours in kNN
use_true_G = false;
k = 9;
dx = 2;
if use_true_G
    G = makeKnnG(data.laloel(1:2, subSampleInd), k);
else
    G = makeKnnG(Y, k);
    %
    % Gaussian kernel. Independent of k.
    %sum2Y = sum(Y.^2, 1);
    %D2 = bsxfun(@plus, sum2Y', sum2Y) - 2*(Y'*Y);
    %D2(abs(D2) <= 1e-5) = 0;
    %med = meddistance(Y);
    %G = exp(-D2./(2*med^2/2)) - eye(n);
    %G(G <= 1e-4) = 0;

    %geodesic distance using Dijkstra
    %D_dijk = dijkstra(sqrt(D2), 1:n);
    %meand = mean(D_dijk(:));
    %G = exp(-D_dijk/(2*meand/2));
    %G(G <= 1e-4) = 0;
    
    % random continuous connection
    %D2 = rand(n, n);
    %D2 = D2+D2';
    %G = D2;
    %G(1:(n+1):end) = 0;
    %
    
    %random binary connection
    %G = rand(n) < 0.001;
    %G = (G | G') - eye(n);
    
    %G = ones(n) - eye(n);
end


%% options to lllvm. Include initializations
op = struct();
op.seed = seed;
op.max_em_iter = 20;
op.abs_tol = 1e-1;
op.G = G;
op.dx = dx;
op.alpha0 = 0.1;
op.beta0 = 0.1;
op.gamma0 = 0.1;
%recorder = create_recorder('print_struct');
store_every_iter = 2;
only_cov_diag = true;
recorder = create_recorder_store_latent(store_every_iter, only_cov_diag);
op.recorder = recorder;

% initial value of the posterior covariance cov_x of X. Size: n*dx x n*dx.
% Appear in q(x) = N(x | mean_x, cov_x) where x is the vectorized n*dx x 1 vector.
%
L = diag(sum(G, 1)) - G ;
invOmega = kron(2*L, eye(dx));
invPi = op.alpha0*eye(n*dx) + invOmega;
op.cov_x0 = inv(eye(n*dx) + invPi) + eye(n*dx);
%op.cov_x0 = 5*eye(n*dx);

% initial value of the posterior mean of X. Size: n*dx x 1.
% Appear in q(x) = N(x | mean_x, cov_x) where x is the vectorized n*dx x 1 vector.
op.mean_x0 = op.cov_x0*randn(n*dx, 1) ;
%op.mean_x0 = op.cov_x0*gamrnd(0.5, 20, n*dx, 1);
%op.mean_x0 = 10*rand(n*dx, 1) ;
%op.mean_x0 = reshape([data.laloel(2, subSampleInd), data.laloel(1, subSampleInd)], n*dx, 1);

%% initialize mu_x with isomap result 
%isomap_x = isomap(Y', dx, 12)';
%op.mean_x0 = reshape(isomap_x, dx*n, 1);

%% Run lllvm 
[ results, op ] = lllvm_1ep(Y, op);
%mean_c = results.mean_c;
%mean_x = results.mean_x;

% plot lower bounds 
figure;
plot(results.lwbs, 'o-');
set(gca, 'fontsize', 16);
xlabel('EM iterations');
ylabel('variational lower bounds');

%gplvm_proj = gplvm(Y', 2)';

% rec_vars will contains all the recorded variables.
rec_vars = recorder();

% change seed back
rng(oldRng);

%% write all results to a file 
G_prefices = {'', '-trueg'};
true_G_prefix = G_prefices{use_true_G + 1};
fname = sprintf('ushcn-d%s-s%d-k%d-n%d%s.mat', dataName, seed, k, n, true_G_prefix);
fglobal = funcs_global();
fpath = fglobal.scriptSavedFile(fname);
timestamp = clock();

if exist('gplvm_proj', 'var')
    save(fpath, 'timestamp', 'rec_vars', 'results', 'op', 'Y', 'data', ...
        'gplvm_proj', 'k', 'subSampleInd');
else
    save(fpath, 'timestamp', 'rec_vars', 'results', 'op', 'Y', 'data', ...
        'k', 'subSampleInd');
end

% export all variables to the base workspace.
allvars = who;
warning('off','putvar:overwrite');
putvar(allvars{:});

display(op);
display(results);



end

