%% We test the full EM with artificial data
% with the derivation written in Adding_Epsilon_on_L.pdf
% Mijung wrote on Oct 15, 2015

clear;
clc;

n = 400; % total number of datapoints

% select data flag
data_flag = 1; % 3D Gaussian
k = 15;

% data_flag = 3; % swiss roll
% k = 5;

% maximum number of EM iterations to perform
max_em_iter = 10;

% true/false to store result. If true, record all variables updated in every
% EM iteration.
is_results_stored = true;

maxseed = 1;
seedpool = 1:maxseed;

for seednum = 1:maxseed
    
    oldRng = rng();
    rng(seednum);
    display(sprintf('seednum %d out of %d', seednum, maxseed));
    
    %% generate data from the true model
    
    % dx = 2; % dim(x)
    % dy = 3; % dim(y)
    % n = 400;  % number of datapoints
    %
    %
    % % parameters
    % alpha = 10*rand; % precision of X (zero-centering)
    % gamma = 10*rand; % noise precision in likelihood
    %
    % epsilon = 1e-3;
    % howmanyneighbors = 20;
    %
    % [vy, Y, vc, C, vx, X, G,  L, invOmega] = generatedata(dy, dx, n, alpha, gamma, epsilon, howmanyneighbors);
    %
    % invPi = alpha*eye(n*dx) + invOmega;
    
    %%  or artificial data
    
    [n, dy, Y, G, dmat, col, truex] = getartificial(n, data_flag, k);
    
    G = double(G);
    
    Y = reshape(Y,dy,n);
    h = sum(G,2);
    L = diag(h) - G; % laplacian matrix
    
    display(sprintf('rank of L is %3f', rank(L)));
    
    dx = 2;
    invOmega = kron(2*L,eye(dx));
    
    epsilon = 1e-3;
    
    %%
    % plot the raw data
    % figure(200);
    % subplot(221); scatter3( Y(1,:) , Y(2,:) , Y(3,:) , [] , col , 'o', 'filled');
    % subplot(223); scatter(truex(1,:), truex(2,:), 20, col, 'o', 'filled');
    
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
    [Ltilde] = compute_Ltilde(L, epsilon, gamma_new, opt_dec);
    eigv_L = Ltilde.eigL;
%     eigv_L = eig(L);
    
    % check if they match
    %     norm( inv(epsilon*ones(n,1)*ones(1,n) + 2*gamma*L)-Ltilde.Ltilde)
    
    %% EM starts
    while (i_em<=max_em_iter)
        
        
        %% (1) E step
        
        % compute mean and cov of x given c (eq.47 and 48)
        tStart = tic;
        [A, b] = compute_suffstat_A_b(G, mean_c, cov_c, Y, gamma_new, epsilon);
        cov_x = inv(A+ invPi_new);
        mean_x = cov_x*b;
        display(sprintf('E step : q(x) took %3f', toc(tStart)));
        
        tStart = tic;
        % compute mean and cov of c given x (eq.56)
        [Gamma, H, Gamma_L]  = compute_suffstat_Gamma_h(G, mean_x, cov_x, Y, gamma_new, Ltilde);
        cov_c = inv(Gamma + epsilon*J*J' + invOmega);
        mean_c = gamma_new*H*cov_c';
        display(sprintf('E step : q(C) took %3f', toc(tStart)));
        
%         ECTC = dy*cov_c + mean_c' * mean_c; 
%         EXX = cov_x + mean_x * mean_x';
%         
%         subplot(221); imagesc(ECTC);
%         subplot(222); imagesc(EXX);
%         
%         pause; 
        
        %% (2) M step: we don't update hyperparameters. Just compute lower bound with new mean/cov of x and C
        
%         tStart = tic;
        [lwb_likelihood, gamma_new] = exp_log_likeli_update_gamma(mean_c, cov_c, H, Y, L, epsilon, Ltilde,  Gamma_L);
%         display(sprintf('M step : gamma update took %3f', toc(tStart)));
        
%         tStart = tic;
        lwb_C = negDkl_C(mean_c, cov_c, invOmega, J, epsilon);
%         display(sprintf('M step : KL_C took %3f', toc(tStart)));
        
%         tStart = tic;
        [lwb_x, alpha_new] = negDkl_x_update_alpha(mean_x, cov_x, invOmega, eigv_L);
%         display(sprintf('M step : alpha update took %3f', toc(tStart)));
        
        %% (2.half) update invPi using the new alpha
        
        invPi_new = alpha_new*eye(n*dx) + invOmega;
        
        %% (3) compute the lower bound
        
        variational_lwb(i_em) = lwb_likelihood + lwb_C + lwb_x; % eq.(21)+(22)+(23)
        display(sprintf('EM it. %d/%d. lwb = %.3f', i_em, max_em_iter, variational_lwb(i_em)));
        
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
        
        
        figure(1);
        subplot(211); plot(1:i_em, variational_lwb(1:i_em), 'o-');
        reshaped_mean_x = reshape(meanXmat(:,i_em), dx, []);
        subplot(212); scatter(reshaped_mean_x(1,:), reshaped_mean_x(2,:), 20, col, 'o', 'filled');

        figure(2);
        plotlearning(dx,dy,n,reshape(meanCmat(:,i_em),dy,n*dx),Y);
        
%         figure(3);
%         subplot(211); scatter(truex(1,:), truex(2,:), 20, col, 'o', 'filled');
%         reshaped_mean_x = reshape(meanXmat(:,i_em), dx, []);
%         subplot(212); scatter(reshaped_mean_x(1,:), reshaped_mean_x(2,:), 20, col, 'o', 'filled'); 
        
        i_em = i_em + 1;
        
    end
        
    % this should be always 0, if lower bound doens't decrease
    lwb_detector = sum((diff(variational_lwb)<0)&(abs(diff(variational_lwb))>1e-3));
    display(sprintf('# decreasing lwb pts  : %.3f', lwb_detector));
    
    % saving results 
    save(strcat('dataflag ', num2str(data_flag), 'seednum ', num2str(seednum), 'k ', num2str(k), '.mat'), 'variational_lwb', 'meanCmat', 'meanXmat', 'alphamat', 'gammamat', 'Y', 'G', 'col', 'truex', 'lwb_detector');
    
    % change seed back
    rng(oldRng);
    
end

% save lwb_detector_alpha_beta lwb_detector_alpha_beta

%%
% 
% load dataflag1seednum1.mat
% % % load dataflag3seednum1.mat
% % 
% dy = 3;
% dx = 2; 
% n = size(G,1);
% % 
% % figure(1);
% % plot(variational_lwb)
% % 
% figure(2);
% which = 3;
% plotlearning(dx,dy,n,reshape(meanCmat(:,which),dy,n*dx),Y);
% 
% figure(3);
% subplot(211); scatter(truex(1,:), truex(2,:), 20, col, 'o', 'filled');
% reshaped_mean_x = reshape(meanXmat(:,which), dx, []);
% subplot(212); scatter(reshaped_mean_x(1,:), reshaped_mean_x(2,:), 20, col, 'o', 'filled');


% subplot(212);
% plotlearning(dx,dy,n,reshape(mean_c,dy,n*dx),Y);
% figure(2);
% subplot(211);
% plot([vc(:) mean_c(:)]);
% subplot(212);
% plot([vx mean_x]);

