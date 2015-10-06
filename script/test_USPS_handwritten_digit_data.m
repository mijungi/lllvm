function test_USPS_handwritten_digit_data(seed, k, dx)

oldRng = rng();
% seed = 10;
% k = 10;
rng(seed);

%startup
%%
% data downloaded from http://www.gaussianprocess.org/gpml/data/
load ../real_data/usps_resampled.mat

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
num_pt_each_digit = 120; 
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

h = sum(G,2);
L = diag(h) - G; % laplacian matrix
invOmega = kron(2*L,eye(dx));

epsilon_1 = 1e-3; % constant term added to prior precision for C
epsilon_2 = 1e-3; % constant term added to likelihood for y

%% (2) EM

diagU = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (1) initialization of hyperparameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha_init = 2*rand;
invPi_init = alpha_init*eye(size(invOmega)) + invOmega;
if diagU
    beta_init = 2*rand;
    invU_init =  beta_init*eye(dy);
else
    invU_init = diag(rand(dy,1));
end

gamma_init = rand;
invV_init = gamma_init*eye(dy);

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % (1) initialization for mean_x and cov_x to random values
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cov_x_qC = inv(speye(size(invPi_init)) + invPi_init);
mean_x_qC = cov_x_qC*randn(n*dx,1);

% Wittawat: create const struct containing all constants (do not change over EM
% iterations) needed.
const = struct();
% eigenvalues of L. nx1 vector.
const.dx = dx;
const.dy = dy;
const.eigv_L = eig(L);
const.n = n;

maxcount = 20;
icount = 1;

variational_lwb = zeros(maxcount, 1);

% to store results every 10th iteration
meanCmat = zeros(n*dx*dy, maxcount/10);
meanXmat = zeros(n*dx, maxcount/10);
covCmat = zeros(n*dx, maxcount/10);
covXmat = zeros(n*dx, maxcount/10);
alphamat = zeros(maxcount/10,1);
betamat = zeros(maxcount/10,1);
gammamat  = zeros(maxcount/10,1);

%%
while (icount<=maxcount)
    
    fprintf(['Performing: ' num2str(icount) ' of ' num2str(maxcount) ' iterations\n'])
    
    %% (1) E step
    
%     fprintf('E-step')
%     tic;
    % 1.1 : compute mean and cov of c given x (eq.19)
    [invGamma_qX, H_qX] = compute_invGamma_qX_and_H_qX(G, mean_x_qC, cov_x_qC, Y, n, dy, dx);
    if diagU
        beta_init = invU_init(1,1);
        gamma_init =  invV_init(1,1);
        cov_c_qX = inv(gamma_init*invGamma_qX + beta_init*(epsilon_1*eye(size(invOmega)) + invOmega));
        mean_c_qX = invV_init*H_qX*cov_c_qX';
    else
        cov_c_qX = inv(kron(invGamma_qX, invV_init) + kron(epsilon_1*eye(size(invOmega)) +invOmega, invU_init));
        mean_c_qX = cov_c_qX*reshape(invV_init*H_qX, [],1);
    end
    
    % 1.2 : compute mean and cov of x given c (eq.16)
    [invA_qC, B_qC, invA_qC_without_gamma] = compute_invA_qC_and_B_qC(G, mean_c_qX, cov_c_qX, invV_init, Y, n, dy, dx, diagU);
    cov_x_qC = inv(invA_qC + invPi_init);
    mean_x_qC = cov_x_qC*B_qC;
    
%     tt = toc;
%     display(sprintf(' took %.3g seconds', tt));
    
    %% (2) M step
    
%     fprintf('M-step')
%     tic;
    [lwb_C , ~, newU] = Mstep_updateU(invU_init, mean_c_qX, cov_c_qX, invOmega, n, dx, dy, diagU, epsilon_1);
    [newAlpha, lwb_x] = Mstep_updateAlpha(const, invOmega, mean_x_qC, cov_x_qC);
    [newGamma, lwb_likelihood] = Mstep_updateGamma(const, mean_c_qX, cov_c_qX, invGamma_qX, H_qX, n, dy, L, invV_init, diagU, epsilon_2, Y);
    
%     display([newAlpha, newGamma]);
    
%     tt = toc;
%     display(sprintf(' took %.3g seconds', tt));
    
    % update parameters
    invU_init = inv(newU);
    invV_init = newGamma*eye(dy);
    invPi_init =  newAlpha*eye(size(invOmega)) + invOmega;
    
    %% (3) compute the lower bound
    
    variational_lwb(icount) = lwb_likelihood + lwb_C + lwb_x; % eq.(21)+(22)+(23)
 
%     subplot(221); plot(1:icount, variational_lwb(1:icount), 'o-');
%     reshaped_mean_x = reshape(mean_x_qC, dx, []);
%     num_pt_visu = size(Y,2);
%     subplot(222); scatter(reshaped_mean_x(1,1:num_pt_visu), reshaped_mean_x(2,1:num_pt_visu), 20, permuted_val(1:num_pt_visu), 'o', 'filled');
%     pause(0.5);
    
    % store results in every 10th iteration (otherwise the datasize to
    % store is too large).
    if rem(icount, 10)==0
        fprintf('Storing results \n')
        meanCmat(:,icount/10) = mean_c_qX(:);
        meanXmat(:,icount/10) = mean_x_qC(:);
        covCmat(:,icount/10) = diag(cov_c_qX); % store only diag of cov, due to too large size!
        covXmat(:,icount/10) = diag(cov_x_qC); % store only diag of cov, due to too large size!
        alphamat(icount/10) = newAlpha;
        betamat(icount/10) = invU_init(1,1);
        gammamat(icount/10) = newGamma;
    end
    
    icount = icount + 1;
    
end

funcs = funcs_global(); 
filename = ['USPS_k=' num2str(k) '_dx=' num2str(dx) '_n=' num2str(n) '_s=' num2str(seed)];
filename_with_directory = funcs.scriptSavedFile([filename '.mat']);
save(filename_with_directory, 'variational_lwb', 'meanCmat', 'meanXmat', 'alphamat', 'betamat', 'gammamat', 'covCmat', 'covXmat');

% change seed back
rng(oldRng);

% %% options to lllvm. Include initializations
% op = struct();
% op.seed = seed;
% op.max_em_iter = 20;
% % absolute tolerance of the increase of the likelihood.
% % If (like_i - like_{i-1} )  < abs_tol, stop EM.
% op.abs_tol = 1e-1;
% op.G = G;
% op.dx = dx;
% % The factor to be multipled to an identity matrix to be added to the graph 
% % Laplacian. This must be positive and typically small.
% op.ep_laplacian = 1e-3;
% % Intial value of alpha. Alpha appears in precision in the prior for x (low
% % dimensional latent). This will be optimized in M steps.
% op.alpha0 = 1;
% % initial value of beta 
% op.beta0 = 1;
% % initial value of gamma.  V^-1 = gamma*I_dy where V is the covariance in the
% % likelihood of  the observations Y.
% op.gamma0 = 1;
% %recorder = create_recorder('print_struct');
% store_every_iter = 5;
% only_cov_diag = false;
% recorder = create_recorder_store_latent(store_every_iter, only_cov_diag);
% op.recorder = recorder;
% 
% % initial value of the posterior covariance cov_x of X. Size: n*dx x n*dx.
% % Appear in q(x) = N(x | mean_x, cov_x) where x is the vectorized n*dx x 1 vector.
% L = diag(sum(G, 1)) - G + op.ep_laplacian*eye(n);
% invOmega = kron(2*L, eye(dx));
% invPi = op.alpha0*eye(n*dx) + invOmega;
% op.cov_x0 = inv(eye(n*dx) + invPi) + eye(n*dx);
% 
% % initial value of the posterior mean of X. Size: n*dx x 1.
% % Appear in q(x) = N(x | mean_x, cov_x) where x is the vectorized n*dx x 1 vector.
% op.mean_x0 = op.cov_x0*randn(n*dx, 1) ;
% 
% %% Run lllvm 
% [ results, op ] = lllvm(Y, op);
% % filename = ['USPS_k_' num2str(k) '_s_' num2str(seed)];
% % filename = ['USPS_k_' num2str(k) '_s_' num2str(seed) '_PCA_initialisation'];
% 
% % rec_vars will contains all the recorded variables.
% rec_vars = recorder();
% 
% funcs = funcs_global(); 
% filename = ['USPS_k=' num2str(k) '_dx=' num2str(dx) '_n=' num2str(n) '_s=' num2str(seed)];
% filename_with_directory = funcs.scriptSavedFile([filename '.mat']);
% save(filename_with_directory, 'rec_vars', 'results');
% % save filename_with_directory rec_vars  -v7.3;
% % -v7.3
% % change seed back
% rng(oldRng);

