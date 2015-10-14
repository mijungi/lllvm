%% We first test E step
% with the derivation written in Adding_Epsilon_on_L.pdf
% Mijung wrote on Oct 13, 2015
 
clear all;
clc;

oldRng = rng();
seed = 21;
rng(seed);

%% (0) define essential quantities

dx = 2; % dim(x)
dy = 4; % dim(y)
n = 20;  % number of datapoints

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

[vy, Y, vc, C, vx, X, G,  L, invOmega] = generatedata(dy, dx, n, alpha, gamma, epsilon); 

invPi = alpha*eye(n*dx) + invOmega; 

%% (2) EM

% initialization for mean_c and cov_c to random values
cov_c = eye(dx*n) ;
mean_c = randn(dy, dx*n); 

i_em = 1; % first EM iteration number 

variational_lwb = zeros(max_em_iter, 1); % storing lowerbound in each EM iteration.
thresh_lwb = 0.01; % we stop EM when change in lwb gets below this threshold. 

% to store results
if is_results_stored
    
    meanCmat = zeros(n*dx*dy, max_em_iter);
    meanXmat = zeros(n*dx, max_em_iter);
    covCmat = zeros(n*dx, max_em_iter);
    covXmat = zeros(n*dx, max_em_iter);
    
    alphamat = zeros(max_em_iter,1);
    gammamat  = zeros(max_em_iter,1);
    
    % partial lower bound from the likelihood term
    LB_like = zeros(max_em_iter, 1);
    
    % partial lower bound from -KL(q(C)||p(C))
    LB_C = zeros(max_em_iter, 1);
    
    % partial lower bound from -KL(q(x)||p(x))
    LB_x = zeros(max_em_iter, 1);
end

% since now we're only updating mean_c, mean_x, and cov_c, cov_x
% we initialise gamma and alpha with their true values.
% when we do Mstep as well, randomly initialise these. 
new_gamma = gamma;
new_alpha = alpha; 

J = kron(ones(n,1), eye(dx));

%% EM starts
while (i_em<=max_em_iter)
    
    display(sprintf('EM iteration %d/%d', i_em, max_em_iter));
    
    %% (1) E step
    
    % compute mean and cov of x given c (eq.47 and 48)
    [A, b] = compute_suffstat_A_b(G, mean_c, cov_c, Y, new_gamma, epsilon);
    cov_x = inv(A+ invPi);
    mean_x = cov_x*b;

    % compute mean and cov of c given x (eq.56)
    [Gamma, H] = compute_suffstat_Gamma_h(G, mean_x, cov_x, Y, gamma, epsilon); 
    cov_c = inv(Gamma + epsilon*J*J' + invOmega); 
    mean_c = gamma*H*cov_c';
   
    %% (2) M step: we don't update hyperparameters. Just compute lower bound with new mean/cov of x and C
    
    % NOTE: optimisation routine replace with pluging-in estimator. Change this  later. (for alpha and gamma)
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
        alphamat(i_em) = new_alpha;
        gammamat(i_em) = new_gamma;
        
        LB_like(i_em) = lwb_likelihood;
        LB_C(i_em) = lwb_C;
        LB_x(i_em) = lwb_x;
        
    end
    
    i_em = i_em + 1;
    
end

% change seed back
rng(oldRng);

