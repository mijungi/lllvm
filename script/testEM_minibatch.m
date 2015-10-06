%% test EM  with subsets of data during training 

oldRng = rng();
seed = 1;
rng(seed);

% startup
%% (0) define essential quantities

dx = 2; % dim(x)
dy = 50; % dim(y)  
n = 200;  % number of datapoints
% maximum number of EM iterations to perform
max_em_iter = 100;

% parameters  
alpha = 2; % precision of X (zero-centering)

% decide whether U is diagonal or not
diagU = true ; % either 1 or 0

if diagU
    beta = 4; 
    invU = beta*eye(dy);
else
    invU = rand(dy);
    invU = invU + invU';
    invU = invU + dy*eye(dy); % prior cov for C (eq.6) 
end

gamma = 2; % noise precision in likelihood

epsilon_1 = 1e-3; % constant term added to prior precision for C
epsilon_2 = 1e-3; % constant term added to likelihood for y

%% (1) generate data

[vy, Y, vc, C, vx, X, G, invOmega, invV, invU, L, mu_y, V_y] = generatedata_Uc(dy, dx, n, alpha, invU, gamma, epsilon_1, epsilon_2);

%% (2) EM using minimatch 

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
    
%% EM starts 
while (i_em<=max_em_iter)
    
    [i_em max_em_iter]
    
    %% (1) E step
        
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
    
    %% (2) M step 

    [lwb_C , ~, newU] = Mstep_updateU(invU_init, mean_c_qX, cov_c_qX, invOmega, n, dx, dy, diagU, epsilon_1); 

    [newAlpha, lwb_x] = Mstep_updateAlpha(const, invOmega, mean_x_qC, cov_x_qC);

    [newGamma, lwb_likelihood] = Mstep_updateGamma(const, mean_c_qX, cov_c_qX, invGamma_qX, H_qX, n, dy, L, invV_init, diagU, epsilon_2, Y);

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

    i_em = i_em + 1; 
    
end

%% sanity check
% check if the first two moments of y generated from estimated mean, cov,
% params match the ones in original data

% (1) compute mean and cov of likelihood in eq. 8
E_estimates = compute_E(G, invV_init, mean_c_qX, mean_x_qC, dx, dy, n); 
% prec_Yest = kron(2*L, invV_init);
% cov_yest = inv(prec_Yest);
% mu_yest = prec_Yest\E_estimates;

%prec_Yest = epsilon_2*eye(n*dy) + kron(2*L, invV_init);
%cov_yest = inv(prec_Yest);
%
% This assumes that invV_init is a scaled identity.
cov_yest = kron(inv(epsilon_2*eye(n) + 2*L*invV_init(1, 1)), eye(dy));
mu_yest = cov_yest*E_estimates;

% (2) compare them to mu_y and cov_y
figure(1);
subplot(211); 
plot(1:dy*n, mu_y(:), 'k', 1:dy*n, mu_yest, 'r'); 
title('mean comparison');

subplot(212); 
cov_y = kron(V_y, eye(dy));
plot(1:dy*n, diag(cov_y), 'k', 1:dy*n, diag(cov_yest), 'r'); 
title('cov comparison');

% (3) look at the estimates
figure(2); 
subplot(221); 
plotlearning(dx,dy,n,reshape(vc,dy,n*dx),Y);
subplot(222); 
plotlearning(dx,dy,n,reshape(mean_c_qX,dy,n*dx),Y);
subplot(223); 
plot([vc(:) mean_c_qX(:)]);
subplot(224); 
plot([vx mean_x_qC]);

% change seed back
rng(oldRng);
