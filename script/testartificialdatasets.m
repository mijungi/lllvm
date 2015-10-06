%%  test artificial datasets on the new model with diagonal U

clear all;
clc;
close all

% startup

% for Swiss roll example, I used k=6, n=800, seed = 1. 
% oldRng = rng();
% seed = 1;
% rng(seed);
% n = 800;
% k = 6;
% data_flag = 3; % swiss roll
% [n, dy, Y, G, dmat, col, truex] = getartificial(n, data_flag, k);

% for 3D Gaussian example, I used 
% oldRng = rng();
% seed = 1;
% rng(seed);
% data_flag = 1; 
% n = 800;
% k = 20;
% [n, dy, Y, G, dmat, col, truex] = getartificial(n, data_flag, k);
% 
% dx = 2;

% for Swiss roll example with a hole
oldRng = rng();
seed = 1;
rng(seed);
data_flag = 2; 
n = 800;
k = 8;
[n, dy, Y, G, dmat, col, truex] = getartificial(n, data_flag, k);

dx = 2;

%%
% plot the raw data
Y = reshape(Y,dy,n);

figure(200);
subplot(221); scatter3( Y(1,:) , Y(2,:) , Y(3,:) , [] , col , 'o', 'filled');
subplot(223); scatter(truex(1,:), truex(2,:), 20, col, 'o', 'filled');

% % (2) GP-LVM
% mappedX_gplvm = gplvm(Y', dx);
% subplot(222); scatter(mappedX_gplvm(:,1), mappedX_gplvm(:,2), 20, col, 'o', 'filled');
% title('gplvm');
% 
% % (3) ISOMAP
% [mappedX_isomap, mapping_isomap] = isomap(Y', dx, k);
% subplot(223); scatter(mappedX_isomap(:,1), mappedX_isomap(:,2), 20, col, 'o', 'filled');
% title('isomap');
% 
% % (4) LLE
% [mappedX_lle, mapping_lle] = lle(Y', dx, k);
% subplot(224); scatter(mappedX_lle(:,1), mappedX_lle(:,2), 20, col(mapping_lle.conn_comp), 'o', 'filled');
% title('LLE');

h = sum(G,2);
epsilon = 1e-3; 
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
alpha_init = 2*rand; 
invPi_init = alpha_init*eye(size(invOmega)) + invOmega;
if diagU
    beta_init = 0.1*rand; 
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
    
    figure(200);
    subplot(222); plot(1:icount, variational_lwb(1:icount), 'o-');
    reshaped_mean_x = reshape(mean_x_qC, dx, []);
    subplot(224); scatter(reshaped_mean_x(1,:), reshaped_mean_x(2,:), 20, col, 'o', 'filled'); title('LLLVM');
    figure(202);
    plotlearning_with_color_code(dx,dy,n,reshape(mean_c_qX, dy, []),Y, col);
    pause(.01);
    
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


%% save results

%     save(strcat('swissroll with ', num2str(n), 'observations', num2str(iter), '_thIter', num2str(k), '_nearestneighbor', '.mat'), 'variational_lwb', 'meanCmat', 'meanXmat', 'alphamat', 'betamat', 'gammamat');
%         save(strcat('swissroll with ', num2str(n), 'observations', num2str(iter), '_thIter', num2str(k), '_nnn_multipleiter', '.mat'), 'variational_lwb', 'meanCmat', 'meanXmat', 'alphamat', 'betamat', 'gammamat', 'Y', 'G', 'col', 'truex');
%
%     end
%
% end

%% look at results:

%     n = 800;
%     iter = 1;
%     k = 5;
%     dx = 2;
% %     dy = 3;
%
% %     load(strcat('swissroll with ', num2str(n), 'observations', num2str(iter), '_thIter', num2str(k), '_nnn_multipleiter', '.mat'));
%    load(strcat('swissroll with ', num2str(n), 'observations', num2str(iter), '_thIter', num2str(k), '_nearestneighbor', '.mat'));
%
%     % generate data again
%     seed = iter;
%     oldRng = rng();
%     rng(seed);
%
%     data_flag = 3; % 3D Gaussian
%     [n, dy, Y, G, dmat, col, truex] = getartificial(n, data_flag, k);
%
%     for i=100
%
%         %     subplot(221); scatter3( Y(1,:) , Y(2,:) , Y(3,:) , [] , col , 'o', 'filled');
%         figure(1);
%         i
%         subplot(211); plot(variational_lwb(1:i), 'o');
%         reshaped_mean_x = reshape(meanXmat(:,i), dx, []);
%         subplot(212); scatter(reshaped_mean_x(1,:), reshaped_mean_x(2,:), 20, col, 'o', 'filled');
%
%         figure(2);
%         mean_c_qX = meanCmat(:,i);
%         plotlearning(dx,dy,n,reshape(mean_c_qX, dy, []),Y);
%
%         pause(0.05);
%     end
%
%     %% plotting the last figure
%
%     % (1) choose a random x*
%
% %     mean_c_qX = meanCmat(:,end);
% %     plotlearning(dx,dy,n,reshape(mean_c_qX, dy, []),Y);
%
% %     reshaped_mean_x = reshape(meanXmat(:,end), dx, []);
% %     min_x1 = min(reshaped_mean_x(1,:));
% %     min_x2 = min(reshaped_mean_x(2,:));
% %
% %     max_x1 = max(reshaped_mean_x(1,:));
% %     max_x2 = max(reshaped_mean_x(2,:));
%
% %     rand_xstar = [mean(reshaped_mean_x(1,:)) + rand.*(max_x1-min_x1)/4; rand]
% %     rand_xstar = [rand; rand];
% %     rand_xstar = [0.4;1.5];
%     rand_xstar = [-1.5; -4];
%     rand_xstar_normalized = rand_xstar./norm(rand_xstar);
%
% %     clf;
%     figure(3);
% %     subplot(211); scatter(truex(1,:), truex(2,:), 20, col, 'o', 'filled');
% %     subplot(212);
%     scatter(reshaped_mean_x(1,:), reshaped_mean_x(2,:), 20, col, 'o', 'filled');
%     hold on;
%     scatter(rand_xstar(1), rand_xstar(2), 'ko');
%
%     param = 1;
%     tt = (2.5*pi/2)*(1+2*rand_xstar_normalized(1));
%     height = 5+rand_xstar_normalized(2);
% %     tt = (2.5*pi/2)*(1+2*rand(1,N));
% %     height = 10*rand(1,N);
%
%     truey = [0.35*tt.*cos(tt); height; 0.35*param*tt.*sin(tt)];
%
%     % compute the distance between x* and all other datapoints, and choose
%     % the k neighest neighbors
% %     mean_x_including_xstar = [reshaped_mean_x rand_xstar];
%     distance = @(a) sqrt(sum(bsxfun(@minus, a, rand_xstar).^2));
%     computed_distance = distance(reshaped_mean_x);
%
%     [sorted_distance, idx_sort] = sort(computed_distance);
%     idx_selected_neighbors = idx_sort(1:k);
%     selected_neighbors = reshaped_mean_x(:, idx_selected_neighbors);
%
% %     scatter(reshaped_mean_x(1,:), reshaped_mean_x(2,:), 20, col, 'o', 'filled'); hold on;
% %     scatter(rand_xstar(1), rand_xstar(2), 'ko');
%     scatter(selected_neighbors(1,:), selected_neighbors(2,:), 'ko');
%
%     reshaped_mean_C = reshape(mean_c_qX, dy, []);
%     linearmap_selected = zeros(dy, dx, k);
%     for i=1:k
%         ind = idx_selected_neighbors(i);
%         linearmap_selected(:,:,i) = reshaped_mean_C(:, (ind-1)*dx+1: dx*ind);
%     end
%
%     reshaped_Y = reshape(Y, dy, []);
%     neighboring_Y = reshaped_Y(:, idx_selected_neighbors);
%
%     sumall = zeros(dy, k);
%
%     for i=1:k
%         sumall(:,i) = neighboring_Y(:,i) +  linearmap_selected(:,:,i)*(rand_xstar - selected_neighbors(:,i));
%     end
%
%     truevsest = [truey mean(sumall,2) ]
%     est_y = mean(sumall,2);
% %
%
% %   scatter3(truey(1), truey(2), truey(3), 'ro'); hold on;
% %   scatter3(est_y(1), est_y(2), est_y(3), 'ro'); hold on;
% %   set(gca, 'zlim',  [-5 3], 'ylim', [-5 15], 'xlim', [-4 4]); axis off;
%
% % figure;
% %%
% figure(4);
% clf;
% % subplot(211);
% % plotlearning(dx,dy,n,reshape(mean_c_qX, dy, []),Y, col);
%
% col_neigh = col(idx_selected_neighbors);
% % col_neigh = 4*ones(k,1);
% plotlearning(dx,dy,k,reshape(linearmap_selected, dy, []),neighboring_Y(:));
%  hold on; scatter3(est_y(1), est_y(2), est_y(3), 'ro');
%    set(gca, 'zlim',  [-5 3], 'ylim', [-5 15], 'xlim', [-4 4]);
%
% % figure(5);
% %   scatter3(truey(1), truey(2), truey(3), 'ro'); hold on;
% %   scatter3(est_y(1), est_y(2), est_y(3), 'ro'); hold on;
% %   set(gca, 'zlim',  [-5 3], 'ylim', [-5 15], 'xlim', [-4 4]);
% %   plotlearning(dx,dy,k+1,[reshape(linearmap_selected, dy, [])],[neighboring_Y(:)]);
%
%
%
%
%
%
