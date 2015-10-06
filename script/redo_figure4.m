%% re-do figure 4

clear all;
clc;
close all

% startup

% for Swiss roll example 
oldRng = rng();
seed = 1;
rng(seed);
data_flag = 3; 
n = 800;
k = 7;
[n, dy, Y, G, dmat, col, truex] = getartificial(n, data_flag, k);

dx = 2;

%%
% plot the raw data
Y = reshape(Y,dy,n);

% figure(200);
% subplot(221); scatter3( Y(1,:) , Y(2,:) , Y(3,:) , [] , col , 'o'); grid off;
% subplot(223); scatter(truex(1,:), truex(2,:), 20, col, 'o', 'filled');

[sort_val, sort_idx] = sort(col);
idx_blue_pt = sort_idx(100);
idx_red_pt = sort_idx(end-80);

%% show them in 2D.

scatter(truex(1,:), truex(2,:), 10, col, 'o');
hold on;
scatter(truex(1,idx_blue_pt), truex(2,idx_blue_pt), 40, 'bo', 'filled');
scatter(truex(1,idx_red_pt), truex(2,idx_red_pt), 40, 'ro', 'filled');

%% show them in 3D. 

scatter3( Y(1,:) , Y(2,:) , Y(3,:) , [] , col , 'o'); grid off;
hold on; 
scatter3( Y(1,idx_blue_pt) , Y(2,idx_blue_pt) , Y(3,idx_blue_pt) , 'bo', 'filled'); grid off;
scatter3( Y(1,idx_red_pt) , Y(2,idx_red_pt) , Y(3,idx_red_pt) , 'ro', 'filled'); grid off;


%%

% uncomment this for shortcircuiting:
G(idx_blue_pt, idx_red_pt) = 1;
G(idx_red_pt, idx_blue_pt) = 1;

h = sum(G,2);
epsilon = 0.008; 
L = diag(h) - G; % laplacian matrix
% L= L + epsilon*eye(size(L));
invOmega = kron(2*L,eye(dx));

epsilon_1 = epsilon; % constant term added to prior precision for C
epsilon_2 = epsilon; % constant term added to likelihood for y

%% (2) EM

diagU = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (1) initialization of hyperparameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha_init = rand; 
invPi_init = alpha_init*eye(size(invOmega)) + invOmega;
if diagU
    beta_init = rand; 
    invU_init =  beta_init*eye(dy);
else
    invU_init = diag(rand(dy,1));
end
    
gamma_init = rand; 
invV_init = gamma_init*eye(dy);

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % (1) initialization for mean_x and cov_x to random values 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [mappedX, mapping] = lle(Y', dx, k);
% subplot(221); scatter(mappedX(:,1), mappedX(:,2), 20, col, 'o', 'filled');
% mean_x_qC = mappedX(:);
% cov_x_qC = inv(0.001*speye(size(invPi_init)) + invPi_init);
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

maxcount = 40;
icount = 1;

variational_lwb = zeros(maxcount, 1);

% to store results
meanCmat = zeros(n*dx*dy, maxcount);
meanXmat = zeros(n*dx, maxcount);
alphamat = zeros(maxcount,1);
betamat = zeros(maxcount,1);
gammamat  = zeros(maxcount,1);

%%
while (icount<=maxcount)
    
    [icount maxcount]
    
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
    
    variational_lwb(icount) = lwb_likelihood + lwb_C + lwb_x; % eq.(21)+(22)+(23)
    
%     figure(200);
%     subplot(222); plot(1:icount, variational_lwb(1:icount), 'o-');
%     reshaped_mean_x = reshape(mean_x_qC, dx, []);
%     subplot(224); scatter(reshaped_mean_x(1,:), reshaped_mean_x(2,:), 20, col, 'o', 'filled'); title('LLLVM');
%     figure(202);
%     plotlearning_with_color_code(dx,dy,n,reshape(mean_c_qX, dy, []),Y, col);
%     pause(.01);
    
    % store results
    meanCmat(:,icount) = mean_c_qX(:);
    meanXmat(:,icount) = mean_x_qC(:);
    alphamat(icount) = newAlpha;
    betamat(icount) = invU_init(1,1);
    gammamat(icount) = newGamma;
    
    icount = icount + 1;
    
end


% change seed back
rng(oldRng);


%%

reshaped_mean_x = reshape(mean_x_qC, dx, []);
%    -541.7970: without shortcircuiting (epsilon = 0.005)
%  -488.2: with shortcircuiting
scatter(reshaped_mean_x(1,:), reshaped_mean_x(2,:),10,  col, 'o'); 
% title(' -2069.4: without shortcircuiting');
hold on;
scatter(reshaped_mean_x(1,idx_blue_pt), reshaped_mean_x(2,idx_blue_pt), 40, 'bo', 'filled');
scatter(reshaped_mean_x(1,idx_red_pt), reshaped_mean_x(2,idx_red_pt), 40, 'ro', 'filled');



