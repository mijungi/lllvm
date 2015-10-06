%% test EM
% problematic case (lower bound decreases): seed=46, dx=2, dy=4, n=50
% problematic case (lower bound decreases): seed=21, dx=2, dy=4, n=120


oldRng = rng();
seed = 21;
rng(seed);
%clear all;
%clc;

% startup
%% (0) define essential quantities

dx = 2; % dim(x)
dy = 4; % dim(y)
n = 100;  % number of datapoints
% maximum number of EM iterations to perform
max_em_iter = 400;
% true/false to store result. If true, record all variables updated in every
% EM iteration.
is_results_stored = true;

% parameters
alpha = 1e-3; % precision of X (zero-centering)

% decide whether U is diagonal or not
diagU = true ; % either 1 or 0

if diagU
    beta = 1e-2;
    invU = beta*eye(dy);
else
    invU = .2*rand(dy);
    invU = invU + invU';
    invU = invU + 0.1*eye(dy); % prior cov for C (eq.6)
end

gamma = .2; % noise precision in likelihood

epsilon = 1e-4; 
% epsilon_1 = 0; % constant term added to prior precision for C
% epsilon_2 = 0; % constant term added to likelihood for y

%% (1) generate data

% get rid of invPi
% tic;
[vy, Y, vc, C, vx, X, G, invOmega, invV, invU, L, mu_y, V_y] = generatedata_test_epsilon(dy, dx, n, alpha, invU, gamma, epsilon);

%[vy, Y, vc, C, vx, X, G, invOmega, invPi, invV, invU, L, mu_y, cov_y] = generatedata_Uc_old(dy, dx, n, alpha, invU, gamma);

% toc;
% V_y = inv(epsilon_y*eye(N) + 2*L*gamma);

%% check E and M step given true values to make sure each E and M works correctly

% % (1) E step for c
% [invGamma_qX, H_qX] = compute_invGamma_qX_and_H_qX(G, vx, 1e-3*eye(n*dx), Y, n, dy, dx);
%
% cov_c = inv(kron(invGamma_qX, invV) + kron(invOmega, invU));
% mean_c = cov_c*reshape(invV*H_qX, [],1);
%
% % check new formula in terms of MN for diagU==1
% if diagU
%     invcovC = gamma*invGamma_qX + beta*invOmega;
%     invcovc = kron(invcovC, eye(dy));
%
%     % matrix normal posterior q(C) = MN(meanC, I_dy, covC)
%     covC = inv(invcovC);
%     meanC = invV*H_qX*covC';
%
%     subplot(311); imagesc([cov_c  inv(invcovc)])
%     subplot(312); plot([mean_c meanC(:)]);
% end
%
% subplot(313); plot([mean_c vc]);
%
% if diagU
%     [invA_qC, B_qC, invA_qC_without_gamma] = compute_invA_qC_and_B_qC(G, reshape(C, dy, []), 1e-3*eye(n*dx), invV, Y, n, dy, dx, diagU);
% else
%     [invA_qC, B_qC, invA_qC_without_gamma] = compute_invA_qC_and_B_qC(G, vc, 1e-3*eye(n*dx*dy), invV, Y, n, dy, dx, diagU);
% end
%
% cov_x = inv(invA_qC + invPi);
% mean_x = cov_x*B_qC;
%
% subplot(212); plot([mean_x vx]);
%
% % M step
% if diagU
%     [lwb_C , ~, newU] = Mstep_updateU(invU, reshape(vc, dy, []), 1e-3*eye(n*dx), invOmega, n, dx, dy, diagU);
% else
%     [lwb_C , ~, newU] = Mstep_updateU(invU, vc, 1e-3*eye(n*dx*dy), invOmega, n, dx, dy, 0);
% end
%
% [invU inv(newU)]
%
% % eq(23)
% [newAlpha, lwb_x] = Mstep_updateAlpha(invOmega, vx, 1e-3*eye(n*dx));
% [alpha newAlpha]
%
% % eq(21)
% [~, D_without_gamma] = computeD(G, Y, invV, L); % that appears in eq.(24)
% if diagU
%     [newGamma, lwb_likelihood] = Mstep_updateGamma(reshape(vc, dy, []), 1e-3*eye(n*dx), invGamma_qX, H_qX, D_without_gamma, n, dy, L, invV, diagU);
% else
%    [newGamma, lwb_likelihood] = Mstep_updateGamma(vc, 1e-3*eye(n*dx*dy), invGamma_qX, H_qX, D_without_gamma, n, dy, L, invV, 0);
% end
%
% [gamma newGamma]

%% (2) EM

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (1) initialization of hyperparameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha_init = .2*rand;
invPi_init = alpha_init*eye(size(invOmega)) + invOmega;
if diagU
    beta_init = .2*rand;
    invU_init =  beta_init*eye(dy);
else
    invU_init = diag(rand(dy,1));
end

gamma_init = 0.1*rand;
invV_init = gamma_init*eye(dy);

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % (1) initialization for mean_x and cov_x to random values
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% cov_x_qC = inv(speye(size(invPi_init)) + invPi_init);
% mean_x_qC = (cov_x_qC + 1.0*eye(size(cov_x_qC)))*randn(n*dx,1);
%mean_x_qC = 2*randn(n*dx,1);

H_init = rand(dy, dx*n);
if diagU
    cov_c_qX =  inv(gamma_init*eye(dx*n) + beta_init*(epsilon*eye(size(invOmega)) +invOmega));
    mean_c_qX = (invV_init*H_init)*cov_c_qX';
else
    cov_c_qX = inv(kron(eye(dx*n), invV_init) + kron(invOmega, invU_init));
    mean_c_qX = cov_c_qX*reshape(invV_init*H_init, [], 1);
end


i_em = 1;

variational_lwb = zeros(max_em_iter, 1);
thresh_lwb = 0.01;

% Wittawat: create const struct containing all constants (do not change over EM
% iterations) needed.
const = struct();
% eigenvalues of L. nx1 vector.
const.dx = dx;
const.dy = dy;
const.eigv_L = eig(L);
const.n = n;

% to store results
if is_results_stored
    meanCmat = zeros(n*dx*dy, max_em_iter);
    meanXmat = zeros(n*dx, max_em_iter);
    covCmat = zeros(n*dx, max_em_iter);
    covXmat = zeros(n*dx, max_em_iter);
    alphamat = zeros(max_em_iter,1);
    betamat = zeros(max_em_iter,1);
    gammamat  = zeros(max_em_iter,1);
    
    % partial lower bound from the likelihood term
    LB_like = zeros(max_em_iter, 1);
    
    % partial lower bound from the C term
    LB_C = zeros(max_em_iter, 1);
    
    % partial lower bound from the x term
    LB_x = zeros(max_em_iter, 1);
end
%% EM starts
while (i_em<=max_em_iter)
    
    display(sprintf('EM iteration %d/%d', i_em, max_em_iter));
    
    %% (1) E step
    
    
    % 1.2 : compute mean and cov of x given c (eq.16)
    %if mod(i_em, 5)==0
    [invA_qC, B_qC ] = compute_invA_qC_and_B_qC(G, mean_c_qX, cov_c_qX, invV_init, Y, n, dy, dx, diagU);
    %end
    %if mod(i_em, 5) == 0
    %if i_em == 1
    cov_x_qC = inv(invA_qC+ invPi_init);
    mean_x_qC = cov_x_qC*B_qC;
    %end
    
    % 1.1 : compute mean and cov of c given x (eq.19)
    %if mod(i_em, 1)==0
    [invGamma_qX, H_qX] = compute_invGamma_qX_and_H_qX(G, mean_x_qC, cov_x_qC, Y, n, dy, dx);
    %end
    if diagU
        beta_init = invU_init(1,1);
        gamma_init =  invV_init(1,1);
        
        %if i_em == 1
        cov_c_qX = inv(gamma_init*invGamma_qX + beta_init*(epsilon*eye(size(invOmega)) + invOmega));
        mean_c_qX = invV_init*H_qX*cov_c_qX';
        %end
%     else
%         cov_c_qX = inv(kron(invGamma_qX, invV_init) + kron(epsilon_1*eye(size(invOmega)) +invOmega, invU_init));
%         mean_c_qX = cov_c_qX*reshape(invV_init*H_qX, [],1);
    end
    
    %     % 1.2 : compute mean and cov of x given c (eq.16)
    %     %if mod(i_em, 5)==0
    %         [invA_qC, B_qC ] = compute_invA_qC_and_B_qC(G, mean_c_qX, cov_c_qX, invV_init, Y, n, dy, dx, diagU);
    %     %end
    %     %if mod(i_em, 5) == 0
    %     %if i_em == 1
    %         cov_x_qC = inv(invA_qC+ invPi_init);
    %         mean_x_qC = cov_x_qC*B_qC;
    %     %end
    
    %% (2) M step
    
    [lwb_C , ~, newU] = Mstep_updateU(invU_init, mean_c_qX, cov_c_qX, invOmega, n, dx, dy, diagU, epsilon);
    
    [newAlpha, lwb_x] = Mstep_updateAlpha(const, invOmega, mean_x_qC, cov_x_qC);
%     [~, D_without_gamma] = computeD(G, Y, invV_init, L, epsilon_2);
    %[newGamma, lwb_likelihood] = Mstep_updateGamma(const, mean_c_qX, cov_c_qX, invGamma_qX, H_qX, n, dy, L, invV_init, diagU, epsilon_2, Y);
    [newGamma, lwb_likelihood] = Mstep_updateGamma(const, mean_c_qX, cov_c_qX, ...
        invGamma_qX, H_qX, n, dy, L, invV_init, diagU, epsilon, Y);
    
%      Mstep_updateGamma(const, mean_c, cov_c, invGamma, H, n, ny, L, invV, diagU, epsilon_2, y, D_without_gamma)

    
    % update parameters
    invU_init = inv(newU);
    invV_init = newGamma*eye(dy);
    invPi_init =  newAlpha*eye(size(invOmega)) + invOmega;
    
    %% (3) compute the lower bound
    
    variational_lwb(i_em) = lwb_likelihood + lwb_C + lwb_x; % eq.(21)+(22)+(23)
    
    figure(102);
    plot(1:i_em, variational_lwb(1:i_em), 'o-');
    %     subplot(211); plot(1:i_em, variational_lwb(1:i_em), 'o-');
    %     reshaped_mean_x = reshape(mean_x_qC, dx, []);
    %     subplot(212); scatter(reshaped_mean_x(1,:), reshaped_mean_x(2,:), 20, 'o', 'filled');
    %     pause(.01);
    
    % store results (all updated variables)
    if is_results_stored
        meanCmat(:,i_em) = mean_c_qX(:);
        meanXmat(:,i_em) = mean_x_qC(:);
        covCmat(:,i_em) = diag(cov_c_qX); % store only diag of cov, due to too large size!
        covXmat(:,i_em) = diag(cov_x_qC); % store only diag of cov, due to too large size!
        alphamat(i_em) = newAlpha;
        betamat(i_em) = invU_init(1,1);
        gammamat(i_em) = newGamma;
        
        LB_like(i_em) = lwb_likelihood;
        LB_C(i_em) = lwb_C;
        LB_x(i_em) = lwb_x;
        
    end
    
    i_em = i_em + 1;
    
end

%% sanity check
% check if the first two moments of y generated from estimated mean, cov,
% params match the ones in original data

% % (1) compute mean and cov of likelihood in eq. 8
% E_estimates = compute_E(G, invV_init, mean_c_qX, mean_x_qC, dx, dy, n);
% % prec_Yest = kron(2*L, invV_init);
% % cov_yest = inv(prec_Yest);
% % mu_yest = prec_Yest\E_estimates;
% 
% %prec_Yest = epsilon_2*eye(n*dy) + kron(2*L, invV_init);
% %cov_yest = inv(prec_Yest);
% %
% % This assumes that invV_init is a scaled identity.
% cov_yest = kron(inv(epsilon*eye(n) + 2*L*invV_init(1, 1)), eye(dy));
% mu_yest = cov_yest*E_estimates;
% 
% % (2) compare them to mu_y and cov_y
% figure(1);
% subplot(211);
% plot(1:dy*n, mu_y(:), 'k', 1:dy*n, mu_yest, 'r');
% title('mean comparison');
% 
% subplot(212);
% cov_y = kron(V_y, eye(dy));
% plot(1:dy*n, diag(cov_y), 'k', 1:dy*n, diag(cov_yest), 'r');
% title('cov comparison');
% 
% % (3) look at the estimates
% figure(2);
% subplot(221);
% plotlearning(dx,dy,n,reshape(vc,dy,n*dx),Y);
% subplot(222);
% plotlearning(dx,dy,n,reshape(mean_c_qX,dy,n*dx),Y);
% subplot(223);
% plot([vc(:) mean_c_qX(:)]);
% subplot(224);
% plot([vx mean_x_qC]);

% change seed back
rng(oldRng);

