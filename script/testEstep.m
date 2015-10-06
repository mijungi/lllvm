%% to test E step only
% Let's try to be rigorous, shall we?
% Mijung wrote on Sep 30, 2015


% now I checked if my code is the same as derivation.
% go back to derivation, see if anything is wrong there. 
% once I am sure my derivation is correct, go back to code if it matches
% derivation. 

clear all;
clc;

oldRng = rng();
seed = 21;
rng(seed);

%% (0) define essential quantities

dx = 2; % dim(x)
dy = 4; % dim(y)
n = 100;  % number of datapoints

% maximum number of EM iterations to perform
max_em_iter = 100;

% true/false to store result. If true, record all variables updated in every
% EM iteration.
is_results_stored = true;

% parameters
alpha = 1; % precision of X (zero-centering)

diagU = 1; % indicator that U is a diagonal matrix
invU = eye(dy); 
gamma = 1; % noise precision in likelihood

epsilon = 1e-2;

%% (1) generate data

[vy, Y, vc, C, vx, X, G, invOmega, invV, invU, L, mu_y, V_y] = generatedata_test_epsilon(dy, dx, n, alpha, invU, gamma, epsilon);

invPi = alpha*eye(n*dx) + invOmega; 

%% (2) EM
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % (1) initialization for mean_c and cov_c to random values
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cov_c_qX = eye(dx*n) ;
mean_c_qX = randn(dy, dx*n); 

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
const.alpha = alpha; 
const.gamma = gamma; 

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
    
    % compute mean and cov of x given c (eq.16)
    [invA_qC, B_qC] = compute_invA_qC_and_B_qC(G, mean_c_qX, cov_c_qX, invV, Y, n, dy, dx, diagU);
    cov_x_qC = inv(invA_qC+ invPi);
    mean_x_qC = cov_x_qC*B_qC;

    % compute mean and cov of c given x (eq.19)
    [invGamma_qX, H_qX] = compute_invGamma_qX_and_H_qX(G, mean_x_qC, cov_x_qC, Y, n, dy, dx);
    cov_c_qX = inv(gamma*invGamma_qX + epsilon*eye(size(invOmega)) + invOmega);
    mean_c_qX = invV*H_qX*cov_c_qX';
   
    %% (2) M step: we don't update hyperparameters. Just compute lower bound with new mean/cov of x and C
    
    % NOTE: optimisation routine replace with pluging-in estimator. Change this
    % later. (for alpha and gamma)
    [lwb_C] = Mstep_updateU(invU, mean_c_qX, cov_c_qX, invOmega, n, dx, dy, diagU, epsilon);
    [newAlpha, lwb_x] = Mstep_updateAlpha(const, invOmega, mean_x_qC, cov_x_qC);
    [newGamma, lwb_likelihood] = Mstep_updateGamma(const, mean_c_qX, cov_c_qX, invGamma_qX, H_qX, n, dy, L, invV, diagU, epsilon, Y);
    
    %% (3) compute the lower bound
    
    variational_lwb(i_em) = lwb_likelihood + lwb_C + lwb_x; % eq.(21)+(22)+(23)
    
    figure(102);
    plot(1:i_em, variational_lwb(1:i_em), 'o-');
    
    % store results (all updated variables)
    if is_results_stored
        meanCmat(:,i_em) = mean_c_qX(:);
        meanXmat(:,i_em) = mean_x_qC(:);
        covCmat(:,i_em) = diag(cov_c_qX); % store only diag of cov, due to too large size!
        covXmat(:,i_em) = diag(cov_x_qC); % store only diag of cov, due to too large size!
        alphamat(i_em) = newAlpha;
        betamat(i_em) = invU(1,1);
        gammamat(i_em) = newGamma;
        
        LB_like(i_em) = lwb_likelihood;
        LB_C(i_em) = lwb_C;
        LB_x(i_em) = lwb_x;
        
    end
    
    i_em = i_em + 1;
    
end

% change seed back
rng(oldRng);

