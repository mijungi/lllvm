function [invA_qC, B_qC, invA_without_gamma] = compute_invA_qC_and_B_qC(G, mean_c, cov_c, invV, Y, n, dy, dx, diagU)
% Input
%  - cov_c: n*dx x n*dx
%  - mean_c: dx x n 
%  - invV: dy x dy
% Outputs:
%  - invA_qC: n*dx x n*dx
%  - B_qC: n*dx x 1

    [invA_qC, B_qC, invA_without_gamma] = compute_invA_qC_and_B_qC_ahmad(G, mean_c, cov_c, invV, Y, n, dy, dx, diagU);

end

function [invA_qC, B_qC, invA_without_gamma] = compute_invA_qC_and_B_qC_wittawat(G, mean_c, cov_c, invV, Y, diagU)
    [dy, n] = size(Y);
    dx = size(cov_c, 1)/n;
    % assume V^-1 = gamma*I_dy
    gamma = invV(1,1);
    %if ~diagU
    %    error('Only diagU=true is supported');
    %end

end

function [invA_qC, B_qC, invA_without_gamma] = compute_invA_qC_and_B_qC_ahmad(G, mean_c, cov_c, invV, Y, n, dy, dx, diagU)
gamma = invV(1,1);

% wait, in this computation, size(C_Sig) should be dx*n by dx*n

if diagU % if U is a diagonal matrix
    C_Sig = cov_c;
    C_Mu = mean_c;
else    
    cov_c_in2D = cellfun(@(x)sum(sum(x)),mat2cell(cov_c,dy*ones(1,n*dx),dy*ones(1,n*dx)),'un',0);
    C_Sig = cell2mat(cov_c_in2D);
    C_Mu = reshape(mean_c, dy, dx*n);    
end

% n*dx x n*dx
ECC = C_Sig + C_Mu' * C_Mu;

%%
% Full <C>_{QC} matrix
% EC = C_Mu;

% Output matrix
% diagcell is a cell array of size 1xn where each element is a square matrix 
% of size dx x dx representing ECC_i (the diagonal blocks of ECC)
diagcell = cellfun(@(x)ECC(x,x), mat2cell(1:size(ECC,1), 1, dx*ones(1,n)), 'un',0);
% n*dx x n*dx
rowrep = repmat( horzcat(diagcell{:}), n, 1);
% n*dx x n*dx
colrep = repmat( vertcat(diagcell{:}), 1, n);

% size: n*dx x n*dx. This code assumes V^-1 = gamma*I i.e., assume V^-1 is a 
% scaled identity.
invA_without_gamma  = - kron(G, ones(dx,dx)) .* (rowrep+colrep) + ...
    ( kron(G,eye(dx,dx)) * (rowrep+colrep) ) .* kron(eye(n),ones(dx,dx)) ;

invA_qC = gamma*invA_without_gamma;

%% invB_qC

EC = C_Mu;
h = sum(G,2);

% Output matrix
prodm = EC'*reshape(Y,dy,n);
rowblocks = mat2cell( kron(G,ones(dx,1)) .* prodm , dx*ones(1,n),n);
B_qC_without_gamma = ( reshape(sum(cat(3,rowblocks{:}),3),dx*n,1) - ...
    sum(prodm' .* kron(eye(n),ones(1,dx)) * kron(G,eye(dx)))' )...
    - ( sum( kron(G,ones(dx,1)) .* prodm ,2) - ...
    kron(h,ones(dx,1)) .* sum( kron(eye(n),ones(dx,1)) .* prodm ,2) );

B_qC = gamma*B_qC_without_gamma;

end

% [invA_qC, B_qC] = compute_invA_qC_and_B_qC(G, mean_c, cov_c, invV, Y, n, dy, dx);

% compute <invA>_q(c), where
% each block has the following form
% <invA_ij>_q(c) = 2*eta_ij*gamma*(delta_ij-1)*< Cj\trp*Cj >_q(c) + gamma*delta_ij* ...
%              sum_k eta_ik <(Ck\trp*Ck + Ci\trp*Ci)>_q(c)

% and compute <B>_q(c), where
% each block has the following form
% <B_i>_q(c) = gamma*\sum_j eta_ij*(<Cj>_q(c)'*(yi-yj) - <Ci>_q(c)'*(yj-yi))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% this is my original code. but I ended up using Ahmad's fast implementation below.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% reshape mean_c and cov_c for further computation

% C = reshape(mean_c, dy, dx, n); % this is mean of C in a tensor
% 
% secondMoments = zeros(dx, dx, n);
% firstTrm_secondMoments = zeros(dx, dx, n);
% 
% dxny = dx*dy;
% for i=1:n
%     %     [i n]
%     % (1) grab (i,i)th chunk of cov_c
%     ijthchunk_cov_c = cov_c((i-1)*dxny+1:i*dxny, (i-1)*dxny+1:i*dxny);
%     % (2) compute summation across diag of each (a,b)th chunk of ijthchunk_cov_c
%     for a = 1:dx
%         for b = 1:dx
%             firstTrm_secondMoments(a,b,i) = sum(diag(ijthchunk_cov_c((a-1)*dy+1:a*dy, (b-1)*dy+1:b*dy)));
%         end
%     end
%     
%     secondMoments(:,:,i) = firstTrm_secondMoments(:,:,i) + C(:,:,i)'*C(:,:,i);
% end
% 
% %%
% 
% dxn = dx*n;
% gamma = invV(1,1);
% 
% invA_qC = zeros(dxn, dxn);
% invA_without_gamma = zeros(dxn, dxn);
% 
% B_qC = zeros(dxn, 1);
% 
% for i = 1 : n
%     
%     %     [i n]
%     
%     CiCi = secondMoments(:,:,i);
%     Ci = C(:,:,i);
%     
%   %%  
%   
%     secondTrm = zeros(dx, dx, n);
%     secondTrm_without_gamma = zeros(dx, dx, n);
%     
%     for k = 1 : n
%         
%         eta_ik = G(i, k);
%         
%         if eta_ik==1
%             
%             CkCk = secondMoments(:,:,k);
%             
%             secondTrm(:,:,k) = gamma*eta_ik*(CkCk + CiCi);
%             secondTrm_without_gamma(:,:,k) = eta_ik*(CkCk + CiCi);
%             
%         end
%         
%     end
%     
%     sum_secondTrm = sum(secondTrm,3);
%     sum_secondTrm_without_gamma = sum(secondTrm_without_gamma,3);
%     
%     %%
%     
%     Bi = zeros(dx, 1, n);
%     
%     for j = 1 : n
%         
%         CjCj = secondMoments(:,:,j);
%         Cj = C(:,:,j);
%         
%         vi = Y(:,i);
%         vj = Y(:,j);
%         
%         eta_ij = G(i, j);
%         
%         %         if eta_ij==1
%         
%         Bi(:, :, j) = gamma*eta_ij*(Cj'*(vi-vj) - Ci'*(vj-vi));
%         
%         if i==j
%             delta_ij = 1;
%         else
%             delta_ij = 0;
%         end
%         
%         firstTrm = eta_ij*gamma*(CjCj + CiCi);
%         firstTrm_without_gamma = eta_ij*(CjCj + CiCi);
%         
%         invA_qC((i-1)*dx+1:i*dx, (j-1)*dx+1:j*dx) = (delta_ij-1)*firstTrm + delta_ij*sum_secondTrm;
%         invA_without_gamma((i-1)*dx+1:i*dx, (j-1)*dx+1:j*dx) = (delta_ij-1)*firstTrm_without_gamma + delta_ij*sum_secondTrm_without_gamma;
%         
%         %         end
%         
%     end
%     
%     B_qC((i-1)*dx+1:i*dx, 1) = sum(Bi, 3);
%     
% end


%% invA_qC

