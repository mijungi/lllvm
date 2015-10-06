%% to generate data from a given true graph
% note: cov(C) has both Omega and U
% Mijung wrote
% September 18, 2014

function [vy, Y, v, C, X, x, G, invOmega, invPi, invV, invU, L, mu_y, cov_y] = generatedata_Uc_old(n, m, N, alpha, invU, gamma)
%        [vy, Y, vc, C, vx, X, G, invOmega, invPi, invV, invU, L, prec_Y] = generatedata_Uc(ny, nx, n, alpha, invU, gamma);

% inputs
% (1) n: dimension of y
% (2) m: dimension of x
% (3) N: number of datapoints
% (4) alpha, U, gamma

% outputs
% (1) vy: vectorized y
% (2) Y: reshape(vy, n, [])
% (3) v: vectorized C
% (4) C: reshape(v, n, m, N)
% (5) X: vectorized x
% (6) x: reshape(X, m, N)
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

% for numerical stability
% to compute inv(invOmeg) later
L = L + 0.001*eye(size(L));

%% (2) draw samples for x, given the given graph

% alpha = 0.01; % strength of zero-centering of x
% smaller alpha, X are more spread
% larger alpha, X are more close to each other

invOmega = kron(2*L, eye(m));
invPi = alpha*eye(N*m) + invOmega;

% draw samples for x 
MU_x = zeros(1, N*m); 
X = mvnrnd(MU_x, inv(invPi)); % eq(5) 
X = X'; 

x = reshape(X, m, N);

figure;
hold on;
gplot(G,x'), for k=1:size(G,1),text(x(1,k),x(2,k),num2str(k)),end,hold off

%% (3) draw samples for C, under the given graph

% matrix normal distribution on C
% p(C|G) \propto 0.5*Tr(invOmega * C'* invU * C)
% vec(X) \propto N(0, Omega \kron U)

% draw samples for C 
prec_C = kron(invOmega, invU);

MU_C = zeros(1, N*m*n); 

v = mvnrnd(MU_C, inv(prec_C))'; % eq(7) 
C = reshape(v, n, m, N);

%% (4) draw samples for y from C and x 

% y \propto N(mu_y, Sig_y)
% Sig_y = inv(2L \kron invV)
% mu_y = Sig_y E

% E has this functional form 

% gamma = 100; 
invV = gamma*eye(n);

E = generateE(G, reshape(C,n,N*m), invV, X(:));

prec_Y = kron(2*L, invV);
cov_y = inv(prec_Y);
mu_y = prec_Y\E;

vy = mvnrnd(mu_y, cov_y); % eq(8)
vy = vy'; 

Y = reshape(vy, n, []);

%% Ahmad's plotting code 

% plotlearning(m,n,N,reshape(C,n,N*m),Y)

