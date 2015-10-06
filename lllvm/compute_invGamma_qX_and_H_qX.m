function [invGamma_qX, H_qX] = compute_invGamma_qX_and_H_qX(G, mean_x_qC, cov_x_qC, Y, n, ny, nx)
%%
% Input:
%  - mean_x_qC: n*dx x 1
%  - G: n x n
%  - cov_x_qC: n*dx x n*dx
%
% tic; 
% Full <X'X>_{QX} matrix
X_Sig = cov_x_qC;
X_Mu = mean_x_qC;

EXX = X_Sig + X_Mu * X_Mu';

% Full <X>_{QX} matrix
% EX = X_Mu;

% Output matrix
%
h = sum(G,2);
IN = speye(n);
kronGIm = kron(G,eye(nx));
kronINIm = kron(IN,ones(nx,nx)); 

% computing second moments per each chunk whose size is (nx) by (nx) 
diagcell = cellfun(@(x)EXX(x,x), ...
    mat2cell(1:size(EXX,1),1,nx*ones(1,n)), ...
    'un',0);

t1 = ( EXX .* kronINIm ) * kron(h,eye(nx));
blkt1 = mat2cell( t1 ,nx*ones(1,n),nx);
t2 = kronINIm .* (repmat( horzcat(diagcell{:}), n, 1) * kronGIm) ;
t3 = kronINIm .* (EXX * kronGIm);
t4 = kronINIm .* (EXX' * kronGIm);
invGamma_qX = blkdiag(blkt1{:}) +  t2 -  t3- t4;

% toc;
kronGones = kron(G,ones(1,nx));
kronINones = kron(eye(n),ones(1,nx)); 

% Output matrix
Yd = reshape(Y,ny,n);
Xd = reshape(X_Mu,nx,n);
e1 = repmat(Xd, n,1)' .* kronGones - ...
    repmat(reshape(Xd*G,n*nx,1)',n,1) .* kronINones;
e2 = (Yd*G)*( repmat(reshape(Xd,n*nx,1)',n,1) .* kronINones ) ;
e3 = ( repmat(h',ny,1).*Yd  )*( kronINones .* repmat(X_Mu',n,1) );

H_qX = Yd*e1 -  e2 + e3;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% this is my original code. but I ended up using Ahmad's fast implementation below.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (1) compute invGamma_qX
% firstmoments_x = reshape(mean_x_qC, nx, []);
% 
% nxn = nx*n;
% invGamma_qX = zeros(nxn, nxn);
% 
% tic; 
% for i = 1 : n
%     invGamma_ii = zeros(nx, nx, n);
%     
%     xi = firstmoments_x(:,i);
%     cov_xixi = cov_x_qC((i-1)*nx+1:i*nx, (i-1)*nx+1:i*nx);
%     secondmoments_xixi = cov_xixi + xi*xi';
%     
%     for j = 1 : n
%         eta_ij = G(i,j);
%         
%         xj = firstmoments_x(:,j);
%         cov_xixj = cov_x_qC((i-1)*nx+1:i*nx, (j-1)*nx+1:j*nx);
%         cov_xjxi = cov_x_qC((j-1)*nx+1:j*nx, (i-1)*nx+1:i*nx);
%         cov_xjxj = cov_x_qC((j-1)*nx+1:j*nx, (j-1)*nx+1:j*nx);
%         
%         secondmoments_xjxj = cov_xjxj + xj*xj';
%         secondmoments_xixj = cov_xixj + xi*xj';
%         secondmoments_xjxi = cov_xjxi + xj*xi';
%         
%         invGamma_ii(:,:,j) = eta_ij*(secondmoments_xjxj - secondmoments_xixj - secondmoments_xjxi + secondmoments_xixi);
%     end
%     
%     invGamma_qX((i-1)*nx+1:i*nx, (i-1)*nx+1:i*nx) = sum(invGamma_ii,3);
%     
% end
% toc;
% (2) compute H_qX
% H_qX = zeros(ny, nxn);
% 
% for i = 1 : n
%     
%     H_i = zeros(ny, nx, n);
%     xi = firstmoments_x(:,i);
%     
%     for j = 1 : n
%         eta_ij = G(i,j);
%         yj = Y(:,j);
%         yi = Y(:,i);
%         xj = firstmoments_x(:,j);
%         
%         H_i(:,:,j) = eta_ij*(yj-yi)*(xj-xi)'; 
% 
%     end
%     
%     H_qX(:, (i-1)*nx+1:i*nx) = sum(H_i,3);
%     
% end

