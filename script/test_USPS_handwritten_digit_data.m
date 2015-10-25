function test_USPS_handwritten_digit_data(seed, k, dx)

oldRng = rng();
% seed = 10;
% k = 10;
rng(seed);

%startup
%%
% data downloaded from http://www.gaussianprocess.org/gpml/data/
load data/usps_resampled.mat

% combine test/train datasets, and select digits corresponds to 0-4 only
y_raw = [train_patterns test_patterns];
y_raw_label = [train_labels test_labels];

% indices for each digit
idx_0 = y_raw_label(1, :) ==1;
idx_1 = y_raw_label(2, :) ==1;
idx_2 = y_raw_label(3, :) ==1;
idx_3 = y_raw_label(4, :) ==1;
idx_4 = y_raw_label(5, :) ==1;

digit_0 = y_raw(:, idx_0);
digit_1 = y_raw(:, idx_1);
digit_2 = y_raw(:, idx_2);
digit_3 = y_raw(:, idx_3);
digit_4 = y_raw(:, idx_4);

% because we want num_pt_each_digit datapoints for each digit
num_pt_each_digit = 80;
digit_0 = digit_0(:,1:num_pt_each_digit);
digit_1 = digit_1(:,1:num_pt_each_digit);
digit_2 = digit_2(:,1:num_pt_each_digit);
digit_3 = digit_3(:,1:num_pt_each_digit);
digit_4 = digit_4(:,1:num_pt_each_digit);

digit_0_to_4 = [digit_0 digit_1 digit_2 digit_3 digit_4];

% take the first num_pt_each_digit datapoints
non_zero_idx_0 = find(idx_0);
digit_0_idx = non_zero_idx_0(1:num_pt_each_digit);
non_zero_idx_1 = find(idx_1);
digit_1_idx = non_zero_idx_1(1:num_pt_each_digit);
non_zero_idx_2 = find(idx_2);
digit_2_idx = non_zero_idx_2(1:num_pt_each_digit);
non_zero_idx_3 = find(idx_3);
digit_3_idx = non_zero_idx_3(1:num_pt_each_digit);
non_zero_idx_4 = find(idx_4);
digit_4_idx = non_zero_idx_4(1:num_pt_each_digit);

digit_val =  [0*ones(size(digit_0_idx)) ...
    1*ones(size(digit_1_idx))...
    2*ones(size(digit_2_idx))...
    3*ones(size(digit_3_idx))...
    4*ones(size(digit_4_idx))];

permuted_val = digit_val;

Y = digit_0_to_4;

n = size(Y, 2);
dy = size(Y,1);

[G, ~] = makeG(Y, n,dy, k);

%%

G = double(G);

Yraw = Y;
Y = reshape(Y,dy,n);
% subtract the mean
Y = bsxfun(@minus, Y, mean(Y, 2));

h = sum(G,2);
L = diag(h) - G;
display(sprintf('rank of L is %3f', rank(L)));

if rank(L)< n-1
    display('sorry, choose a larger k so that rank(L)=n-1');
%     break;
end

%% options to lllvm_1ep. Include initializations
op = struct();
op.seed = seed;
op.max_em_iter = 20;
% absolute tolerance of the increase of the likelihood.
% If (like_i - like_{i-1} )  < abs_tol, stop EM.
op.abs_tol = 1e-1;
op.G = G;
op.dx = dx;
% The factor to be added to the prior covariance of C and likelihood. This must be positive and typically small.
op.epsilon = 1e-3;
% Intial value of alpha. Alpha appears in precision in the prior for x (low
% dimensional latent). This will be optimized in M steps.
op.alpha0 = rand;
% initial value of gamma.  V^-1 = gamma*I_dy where V is the covariance in the
% likelihood of  the observations Y.
op.gamma0 = rand;
%recorder = create_recorder('print_struct');
store_every_iter = 5;
only_cov_diag = false;
recorder = create_recorder_store_latent(store_every_iter, only_cov_diag);
op.recorder = recorder;

invOmega = kron(2*L,eye(dx));

% initial value of the posterior mean and cov of c
op.cov_c = eye(dx*n) ;
op.mean_c = randn(dy, dx*n);


[results, op ] = lllvm_1ep(Y, op);

%%

funcs = funcs_global();
filename = ['USPS_k=' num2str(k) '_dx=' num2str(dx) '_n=' num2str(n) '_s=' num2str(seed)];
filename_with_directory = funcs.scriptSavedFile([filename '.mat']);
save(filename_with_directory, 'results', 'Yraw', 'permuted_val');

% change seed back
rng(oldRng);


%%
% which = 20;
% xx = results.mean_x(:,which);
% xx = reshape(xx, dx, []);
%  scatter(xx(1,:), xx(2,:), 20, permuted_val, 'o', 'filled');
