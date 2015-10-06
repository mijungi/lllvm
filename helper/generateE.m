function E = generateE(G, C, invV, X)

%% inputs
% (1) G is a N by N adjacency matrix, where N is # datapoints
% (2) C is a n by Nm matrix, where n is a dimensionality of y (observation) 
%     and m is the latent dimension. 
%     Each tangent matrix is stored along column in C.
% (3) invV is a n by noise precision matrix. 
% X is a mN by 1 vector, where each length-m vector is x_i

%% outputs
% E is a Nn by 1 matrix, where e_i is given by
% \sum_{j=1}^N (eta_{ji} C_j invV (x_i - x_j) -eta_{ij} C_i invV (x_j - x_i)) 


%% unpack essential quantities

N = size(G,1); 
[n, Nm] = size(C);

m = Nm/N; % latent dimension

E = zeros(N*n, 1);

for i=1:N
    
    Ei = zeros(n, N);
    
    for j=1:N
        
        Ci = C(:, (i-1)*m +1 : i*m); 
        Cj = C(:, (j-1)*m +1: j*m);
        
        xi = X((i-1)*m+1:i*m, 1);
        xj = X((j-1)*m+1:j*m, 1);
        
        eta_ji = G(j,i);
        eta_ij = G(i,j);
        
        Ei(:,j) =  eta_ji*invV*Cj*(xi-xj) - eta_ij*invV*Ci*(xj-xi);
        
    end
    
        E((i-1)*n+1:i*n, :) = sum(Ei,2);
        
end





