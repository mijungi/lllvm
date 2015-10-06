function [ ] = test_frey_face()
%TEST_FREY_FACE Run LL-LVM on Frey face data.
%
%@author Wittawat. Created 22 May 2015.
%

seed = 2;
oldRng = rng();
rng(seed);

frey = load('frey20_14.mat');
%Name        Size                Bytes  Class     Attributes
%desc        1x29                   58  char                
%ff70      280x1965            4401600  double             

%% subsample size to use 
sub_n = 500;
frey_funcs = funcs_frey_face();
Y = frey_funcs.subsample(frey.ff70, sub_n, seed);
[dy, n] = size(Y);

%% construct the graph
% k = number of neighbours in kNN
k = 10;
G = makeKnnG(Y, k);
dx = 2;

%% options to lllvm. Include initializations
op = struct();
op.seed = seed;
op.max_em_iter = 100;
op.abs_tol = 1e-1;
op.G = G;
op.dx = dx;
op.ep_laplacian = 1e-3;
op.alpha0 = 1;
op.beta0 = 1;
op.gamma0 = 1;
%recorder = create_recorder('print_struct');
store_every_iter = 3;
only_cov_diag = true;
recorder = create_recorder_store_latent(store_every_iter, only_cov_diag);
op.recorder = recorder;

% initial value of the posterior covariance cov_x of X. Size: n*dx x n*dx.
% Appear in q(x) = N(x | mean_x, cov_x) where x is the vectorized n*dx x 1 vector.
%
L = diag(sum(G, 1)) - G + op.ep_laplacian*eye(n);
invOmega = kron(2*L, eye(dx));
invPi = op.alpha0*eye(n*dx) + invOmega;
op.cov_x0 = inv(eye(n*dx) + invPi) + eye(n*dx);
%op.cov_x0 = eye(n*dx);

% initial value of the posterior mean of X. Size: n*dx x 1.
% Appear in q(x) = N(x | mean_x, cov_x) where x is the vectorized n*dx x 1 vector.
op.mean_x0 = op.cov_x0*randn(n*dx, 1) ;

%% Run lllvm 
[ results, op ] = lllvm(Y, op);
%mean_c = results.mean_c;
%mean_x = results.mean_x;

% plot lower bounds 
figure;
plot(results.lwbs, 'o-');
set(gca, 'fontsize', 16);
xlabel('EM iterations');
ylabel('variational lower bounds');

% rec_vars will contains all the recorded variables.
rec_vars = recorder();

% change seed back
rng(oldRng);

%% write all results to a file 
fname = sprintf('frey-s%d-k%d-n%d.mat', seed, k, n);
fglobal = funcs_global();
fpath = fglobal.scriptSavedFile(fname);
timestamp = clock();
save(fpath, 'timestamp', 'rec_vars', 'results', 'op', 'Y');

% export all variables to the base workspace.
allvars = who;
warning('off','putvar:overwrite');
putvar(allvars{:});

display(op);
display(results);



end

