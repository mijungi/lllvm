%% to generate data from a given true graph
% note: cov(C) has both Omega and U
% Mijung wrote
% Oct 13, 2015

function [vy, Y, vc, C, vx, X, G,  L, invOmega] = generatedata(dy, dx, n, alpha, gamma, epsilon, howmanyneighbors)

% inputs
% (1) dy: dimension of y
% (2) dx: dimension of x
% (3) n: number of datapoints
% (4) alpha: prior precision for x
% (5) gamma: noise precision in likelihood 
% (6) epsilon: a constant term added to prior precision for C
% (7) howmanyneighbors : for G

% outputs
% (1) vy: vectorized y (length of dy*n)
% (2) Y: size of (dy, n)
% (3) vc: vectorized C
% (4) C: size of (dy, dx*n)
% (5) vx: vectorized x
% (6) X: size of (dx, n)
% (7) G: adjacency matrix
% (8) L: laplacian matrix
% (9) invOmega : kron(2L, eye(dx))

%% (1) define a graph

% howmanyneighbors = 5;

% adjacency matrix
% G = zeros(n,n,howmanyneighbors-1);
% 
% for i=1:howmanyneighbors-1
%     G(:,:,i) = diag(ones(n-i,1),i);
%     G(:,:,i) = G(:,:,i) + G(:,:,i)';
% end
% 
% G = sum(G,3);

Yfake_for_generating_G = randn(dy, n);
G = makeKnnG(Yfake_for_generating_G, howmanyneighbors); 
% [G, ~] = makeG(Yfake_for_generating_G, n,dy, howmanyneighbors);

h = sum(G,2);
% laplacian matrix
L = diag(h) - G;

%% (2) draw samples for x, given the graph

invOmega = kron(2*L,speye(dx));
X = mnormrnd(zeros(dx, n), 1, inv(alpha*eye(n) + 2*L) );
vx = X(:);

%% (3) draw samples for C, given the graph

J = kron(ones(n,1), eye(dx));
ep_invOmega = epsilon*J*J'+ invOmega;

C = mnormrnd(zeros(dy, dx*n), eye(dy), inv(ep_invOmega) );
vc = C(:);

%% (4) draw samples for y from C and x 

invV = gamma*speye(dy);
ones_n = ones(n,1);

V_y = inv(epsilon*ones_n*ones_n' + 2*L*gamma);
E = generateE(G, C, invV, vx);
E = reshape(E, dy, n);
mu_y = E*V_y;
Y = mnormrnd(mu_y, 1, V_y);
vy = Y(:);
