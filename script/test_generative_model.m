%% test our generative model
% do we need all these hyperparameters?
% alpha : prior precision in x
% epsilon_1: prior precision in C
% beta: prior precision in C across rows
% epsilon_2: precision in y
% gamma: noise precision in y

dx = 2; % dim(x)
dy = 4; % dim(y)
n = 100;  % number of datapoints

%% (1) define a graph

howmanyneighbors = 5;

% adjacency matrix
G = zeros(n,n,howmanyneighbors-1);

for i=1:howmanyneighbors-1
    G(:,:,i) = diag(ones(n-i,1),i);
    G(:,:,i) = G(:,:,i) + G(:,:,i)';
end

G = sum(G,3);

h = sum(G,2);
% laplacian matrix
L = diag(h) - G;
rank(L)

%% 

alpha =1; 
invOmega = kron(2*L,speye(dx));
x = mnormrnd(zeros(dx, n), 1, inv(alpha*eye(n) + 2*L) );

% plot(x')
% [var(x)


