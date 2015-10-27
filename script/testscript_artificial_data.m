%% We test the full EM with artificial data
% Mijung wrote on Oct 23, 2015

clear;
clc;

n = 400; % total number of datapoints
data_flag = 1; % 3D Gaussian
kmat = [5, 10, 20, 40, 80, 160];

% data_flag = 3; % swiss roll
% n = 800; % total number of datapoints
% kmat = [5, 10, 20, 40, 80, 160];

% maximum number of EM iterations to perform
max_em_iter = 30;

maxseed = 10;

for seednum = 1:maxseed
    
    oldRng = rng();
    rng(seednum);
    display(sprintf('seednum %d out of %d', seednum, maxseed));
    
    for kidx = 1:length(kmat)
        
        k = kmat(kidx);
        
        %% generate artificial data
        
        [n, dy, Y, G, dmat, col, truex] = getartificial(n, data_flag, k);
        
        G = double(G);
        dx = 2;
        
        Yraw = Y;
        Y = reshape(Y,dy,n);
        % subtract the mean
        %     Y = bsxfun(@minus, Y, mean(Y, 2));
        
        h = sum(G,2);
        L = diag(h) - G;
        display(sprintf('rank of L is %3f', rank(L)));
        
        if rank(L)< n-1
            display('sorry, choose a larger k so that rank(L)=n-1');
            break;
        end
        
        %% options to lllvm_1ep. Include initializations
        op = struct();
        op.seed = seednum;
        op.max_em_iter = max_em_iter;
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

        % initial value of the posterior mean and cov of c. (optional)
        op.cov_c0 = eye(dx*n) ;
        op.mean_c0 = randn(dy, dx*n);
        
        
        [results, op ] = lllvm_1ep(Y, op);
        
        save(strcat('fixingalpha_dataflag ', num2str(data_flag), 'seednum ', num2str(seednum), 'k ', num2str(k), '.mat'), 'results');

%         save(strcat('dataflag ', num2str(data_flag), 'seednum ', num2str(seednum), 'k ', num2str(k), '.mat'), 'results');
        
    end
    
    % change seed back
    rng(oldRng);
    
end

%%

% load dataflag1seednum1.mat
%
% dy = 3;
% dx = 2;
% n = size(G,1);
%
% figure(2);
% which = 3;
% plotlearning(dx,dy,n,reshape(meanCmat(:,which),dy,n*dx),Y);
%
% figure(3);
% subplot(211); scatter(truex(1,:), truex(2,:), 20, col, 'o', 'filled');
% reshaped_mean_x = reshape(meanXmat(:,which), dx, []);
% subplot(212); scatter(reshaped_mean_x(1,:), reshaped_mean_x(2,:), 20, col, 'o', 'filled');
%

%%

% data_flag = 1;
% seednum = 1;
% k = 20;
% load(strcat('dataflag ', num2str(data_flag), 'seednum ', num2str(seednum), 'k ', num2str(k), '.mat'))
%
% which = 10;
%
% figure(1);
% plotlearning(dx,dy,n,reshape(results.mean_c(:,which),dy,n*dx),Yraw);
%
% figure(2);
% subplot(211); plot(results.lwbs);
% reshaped_mean_x = reshape(results.mean_x(:,which), dx, []);
% subplot(212); scatter(reshaped_mean_x(1,:), reshaped_mean_x(2,:), 20, col, 'o', 'filled');


