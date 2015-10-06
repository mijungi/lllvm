%% to generate data from a given true graph
% note: cov(C) has both Omega and U
% Mijung wrote
% September 18, 2014

function [vy, Y, v, C, X, x, G, invOmega, invV, invU, L, mu_y, V_y] = generatedata_Uc(dy, dx, N, alpha, invU, gamma, epsilon_c, epsilon_y)
% [vy, Y, vc, C, vx, X, G, invOmega, invPi, invV, invU, L, mu_y, cov_y] = generatedata_Uc(dy, dx, dy, alpha, invU, gamma, epsilon_c, epsilon_y);
% 
% inputs
% (1) dy: dimension of y
% (2) dx: dimension of x
% (3) N: number of datapoints
% (4) alpha, U, gamma
% (5) epsilon_c: a constant term added to prior precision for C
% (6) epsilon_y: a constant term added to likelihood for y

% outputs
% (1) vy: vectorized y
% (2) Y: reshape(vy, dy, [])
% (3) v: vectorized C
% (4) C: reshape(v, dy, dx, N)
% (5) X: vectorized x
% (6) x: reshape(X, dx, N)
% (7) G: adjacency matrix

%% (1) define a graph

howmanyneighbors = 5;

% adjacency matrix
G = zeros(N,N,howmanyneighbors-1);

for i=1:howmanyneighbors-1
    G(:,:,i) = diag(ones(N-i,1),i);
    G(:,:,i) = G(:,:,i) + G(:,:,i)';
end

G = sum(G,3);

h = sum(G,2);
% laplacian matrix
L = diag(h) - G;

% L = L + 1e-3*eye(size(L));
Leps =  L + 1e-3*eye(size(L));

%% (2) draw samples for x, given the given graph

% alpha = 0.01; % strength of zero-centering of x
% smaller alpha, X are more spread
% larger alpha, X are more close to each other

% invOmega: n*dx x n*dx
invOmega = kron(2*L,speye(dx));
% invPi = alpha*speye(N*dx) + invOmega;

% draw samples for x 
%MU_x = zeros(1, N*dx); 
%X = mvnrnd(MU_x, inv(invPi)); % eq(5) 
%X = X'; 
%x = reshape(X, dx, N);
x = mnormrnd(zeros(dx, N), 1, inv(alpha*eye(N) + 2*L) );
X = x(:);

% figure;
% hold on;
% gplot(G,x') 
% for k=1:size(G,1)
%     text(x(1,k), x(2,k), num2str(k));
% end
% hold off;

%% (3) draw samples for C, under the given graph

% matrix normal distribution on C
% p(C|G) \propto 0.5*Tr(invOmega * C'* invU * C)
% vec(X) \propto N(0, Omega \kron U)

% draw samples for C 
%prec_C = kron(epsilon_c*eye(size(invOmega)) + invOmega, invU);

U = inv(invU);
ep_invOmega = epsilon_c*eye(size(invOmega)) + invOmega;

%inv_prec_C = kron(inv(ep_invOmega), U);
%MU_C = zeros(1, N*dx*dy); 
%v = mvnrnd(MU_C, inv_prec_C)'; % eq(7) 
%C = reshape(v, dy, dx, N);

% more efficient way to draw from a matrix normal distribution
v = mnormrnd(zeros(dy, dx*N), U, inv(ep_invOmega) );
C = v(:);

%% (4) draw samples for y from C and x 

% y \propto N(mu_y, Sig_y)

% E has this functional form 

% gamma = 100; 
invV = gamma*speye(dy);
%prec_Y = epsilon_y*eye(N*dy) + kron(2*L, invV);
%cov_y = inv(prec_Y);
%
V_y = inv(epsilon_y*eye(N) + 2*L*gamma);
E = generateE(G, reshape(C,dy,N*dx), invV, X(:));
% cov_y = kron(V_y, eye(dy));
% mu_y = cov_y*E;
% Y = mnormrnd(reshape(mu_y, dy, N), 1, V_y);

E = reshape(E, dy, N);
mu_y = E*V_y;
Y = mnormrnd(mu_y, 1, V_y);

vy = Y(:);


%%
% test log likelihood formula (only the squared term in exponent)
% from_formula = - vy'*prec_Y*vy; 
% 
% from_exact = zeros(N,1);
% for i=1:N
%     yi = Y(:, i);
%     Ci = C(:,:,i);
%     xi = x(:,i);
%     
%     frst_trm = epsilon_y*yi'*yi;
%     scnd_trm = zeros(N,1);
%     
%     for j=1:N
% %         j
%        eta = G(i,j);
%         yj = Y(:,j);
%         xj = x(:,j);
%         scnd_trm(j) =  eta*gamma*((yj-yi))'*((yj-yi));
%     end
%     
%     from_exact(i) = frst_trm + sum(scnd_trm);
%     
% end
%         
%         
% [-sum(from_exact) from_formula]

%% Ahmad's plotting code 

% plotlearning(dx,dy,N,reshape(C,dy,N*dx),Y)

