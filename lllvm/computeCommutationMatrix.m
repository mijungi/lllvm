function [Krm, Knq] =  computeCommutationMatrix(m, n, r, q)

% A = rand(m, n);
% B = rand(r, q);

vec = @(a) reshape(a, [], 1); 

if r*m == n*q
    
%     % (1) K_rm
% %     commuMat_rm_outer = zeros(r*m, r*m, r);
%     Krm = zeros(r*m, r*m);
    
%     for i=1:r
        
% %         [i r]
        
%         e_r = zeros(r,1);
%         e_r(i) = 1;
%         e_r = sparse(e_r);
        
%         % first, sum over j
%         commuMat_rm = zeros(r*m, r*m, m);
% %         tic; 
%         for j=1:m
%             e_m = zeros(m, 1);
%             e_m(j) = 1;
%             e_m = sparse(e_m);
            
%             erem = e_r*e_m' ;
%             commuMat_rm(:,:,j) = kron(erem, erem');
%         end
% %         toc; 
% %         summed_commuMat_rm = sum(commuMat_rm,3);
%         Krm = Krm + sum(commuMat_rm,3);
%         % store results to sum over i
% %         commuMat_rm_outer(:,:,i) =  summed_commuMat_rm;
%     end
    
%     Krm = sum(commuMat_rm_outer,3);
    Krm = reshape(kron(vec(speye(r)), speye(m)), r*m, r*m);
    Knq = Krm';
    disp('Krm is transpose of Knq');

else
    
    % % (2) K_rm
    
    % commuMat_nq_outer = zeros(n*q, n*q, n);
    
    % for i=1:n
        
    %     e_n = zeros(n,1);
    %     e_n(i) = 1;
        
    %     % first, sum over j
    %     commuMat_nq = zeros(n*q, n*q, q);
    %     for j=1:q
    %         e_q = zeros(q, 1);
    %         e_q(j) = 1;
            
    %         eneq = e_n*e_q' ;
    %         commuMat_nq(:,:,j) = kron(eneq, eneq');
    %     end
    %     summed_commuMat_nq = sum(commuMat_nq,3);
        
    %     % store results to sum over i
    %     commuMat_nq_outer(:,:,i) =  summed_commuMat_nq;
    % end
    
    % Knq = sum(commuMat_nq_outer,3);

    Knq = reshape(kron(vec(speye(n)), speye(q)), n*q, n*q);
    
end