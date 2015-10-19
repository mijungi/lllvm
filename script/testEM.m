%% We test the full EM
% with the derivation written in Adding_Epsilon_on_L.pdf
% Mijung wrote on Oct 15, 2015

clear all;
clc;

maxseed = 1;
seedpool = 1:maxseed;
% lwb_detector_alpha = zeros(maxseed,1);
lwb_detector_alpha_beta = zeros(maxseed,1);

for seednum = 1:maxseed
    
    oldRng = rng();
    % seednum = 2403;
    
    rng(seednum);
    display(sprintf('seednum %d out of %d', seednum, maxseed));
    
    %% (0) define essential quantities
    
    dx = 2; % dim(x)
    dy = 3; % dim(y)
    n = 100;  % number of datapoints
    
    % maximum number of EM iterations to perform
    max_em_iter = 40;
    
    % true/false to store result. If true, record all variables updated in every
    % EM iteration.
    is_results_stored = true;
    
    % parameters
    alpha = 10*rand; % precision of X (zero-centering)
    gamma = 10*rand; % noise precision in likelihood
    
    epsilon = 1e-3;
    howmanyneighbors = (ceil(n/2*rand))+1;
    
    %% (1) generate data
    
    [vy, Y, vc, C, vx, X, G,  L, invOmega] = generatedata(dy, dx, n, alpha, gamma, epsilon, howmanyneighbors);
    
    invPi = alpha*eye(n*dx) + invOmega;
    
    %% (2) EM
    
    % initialization for mean_c and cov_c to random values
    cov_c = eye(dx*n) ;
    mean_c = randn(dy, dx*n);
    
    % initialization for alpha, and update invPi
    alpha_new = rand;
    invPi_new = alpha_new*eye(n*dx) + invOmega;
    gamma_new = rand;
    
    i_em = 1; % first EM iteration number
    variational_lwb = zeros(max_em_iter, 1); % storing lowerbound in each EM iteration.
    
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
    
    J = kron(ones(n,1), eye(dx));
    
    opt_dec = 1; % using decomposition
    [Ltilde] = compute_Ltilde(L, epsilon, gamma, opt_dec);
    eigv_L = Ltilde.eigL;    % eigv_L = eig(L);
    
    % check if they match
    %     norm( inv(epsilon*ones(n,1)*ones(1,n) + 2*gamma*L)-Ltilde.Ltilde)
    
    %% EM starts
    while (i_em<=max_em_iter)
        
        
        %% (1) E step
        
        % compute mean and cov of x given c (eq.47 and 48)
        tic;
        [A, b] = compute_suffstat_A_b(G, mean_c, cov_c, Y, gamma_new, epsilon);
        toc;
        cov_x = inv(A+ invPi_new);
        mean_x = cov_x*b;
        
        % compute mean and cov of c given x (eq.56)
        tic;
        [Gamma, H] = compute_suffstat_Gamma_h(G, mean_x, cov_x, Y, gamma_new, epsilon);
        toc;
        cov_c = inv(Gamma + epsilon*J*J' + invOmega);
        mean_c = gamma_new*H*cov_c';
        
        %% (2) M step: we don't update hyperparameters. Just compute lower bound with new mean/cov of x and C
        
        %         lwb_likelihood = exp_log_likeli(mean_c, cov_c, Gamma, H, Y, L, gamma, epsilon);
        EXX = cov_x + mean_x * mean_x';
        tic;
        [lwb_likelihood, gamma_new] = exp_log_likeli_update_gamma(mean_c, cov_c, Gamma, H, Y, L, gamma_new, epsilon, Ltilde, G, EXX);
        toc;
        
        tic;
        lwb_C = negDkl_C(mean_c, cov_c, invOmega, J, epsilon);
        toc;
        tic;
        [lwb_x, alpha_new] = negDkl_x_update_alpha(mean_x, cov_x, invOmega, eigv_L);
        toc;
        
        %% (2.half) update invPi using the new alpha
        invPi_new = alpha_new*eye(n*dx) + invOmega;
        
        
        %% (3) compute the lower bound
        
        variational_lwb(i_em) = lwb_likelihood + lwb_C + lwb_x; % eq.(21)+(22)+(23)
        display(sprintf('EM it. %d/%d. lwb = %.3f', i_em, max_em_iter, variational_lwb(i_em)));
        
        %figure(102);
        %plot(1:i_em, variational_lwb(1:i_em), 'o-');
        
        % store results (all updated variables)
        if is_results_stored
            
            meanCmat(:,i_em) = mean_c(:);
            meanXmat(:,i_em) = mean_x(:);
            covCmat(:,i_em) = diag(cov_c); % store only diag of cov, due to too large size!
            covXmat(:,i_em) = diag(cov_x); % store only diag of cov, due to too large size!
            
            alphamat(i_em) = alpha_new;
            gammamat(i_em) = gamma_new;
            
            LB_like(i_em) = lwb_likelihood;
            LB_C(i_em) = lwb_C;
            LB_x(i_em) = lwb_x;
            
        end
        
        i_em = i_em + 1;
        
    end
    
    % this should be always 0, if lower bound doens't decrease
    lwb_detector_alpha_beta(seednum) = sum((diff(variational_lwb)<0)&(abs(diff(variational_lwb))>1e-3));
    
    display(sprintf('# decreasing lwb pts  : %.3f', lwb_detector_alpha_beta(seednum)));
    
    % change seed back
    rng(oldRng);
    
end

% save lwb_detector_alpha_beta lwb_detector_alpha_beta

%%

figure(1);
subplot(211);
plotlearning(dx,dy,n,reshape(vc,dy,n*dx),Y);
subplot(212);
plotlearning(dx,dy,n,reshape(mean_c,dy,n*dx),Y);
figure(2);
subplot(211);
plot([vc(:) mean_c(:)]);
subplot(212);
plot([vx mean_x]);

