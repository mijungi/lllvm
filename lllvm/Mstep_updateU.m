function [obj, dobjdU, newU] = Mstep_updateU(invU, mean_c, cov_c, invOmega, n, nx, ny, diagU, epsilon_1)
%function [obj, dobjdU, newU] = Mstep_updateU(invU, mean_c, cov_c, invOmega, n, nx, ny)
%
% update U (the covariance across latent dimensions in C)
%
% Inputs:
%  mean_c: dy x n*dx
%  cov_c: n*dx x n*dx
%  invOmega: n*dx x n*dx


if diagU % if U is a diagonal matrix, we just update beta , where U^{-1} = beta * I _ dy
    
    %% (1) function value
    
    invOmega = epsilon_1*speye(size(invOmega)) + invOmega; 
    
    beta = invU(1,1);     
    
    %nytraceTrm = ny*trace(invOmega*cov_c);
%     % invOmega is symmetric.
    nytraceTrm = ny*invOmega(:)'*cov_c(:);

%     % n*dx x n*dx
    mean_c2 = mean_c'*mean_c;
    %traceTrm = trace(invOmega*(mean_c'*mean_c));
    traceTrm = invOmega(:)'*mean_c2(:);
    
    
    obj_frstTrm = ny*logdetns(cov_c*invOmega) + n*nx*ny*log(beta);
    if abs(imag(obj_frstTrm ) ) > 1e-6
        % first term has an imaginary part. 
        error('obj_firstTrm has an imaginary part. Possibly invOmega is low rank because L is low rank. Add a scaled identity to L.');
    end
    %     obj_scndTrm = ny*beta*trace(invOmega*cov_c) - n*nx*ny;
    obj_scndTrm = beta*nytraceTrm - n*nx*ny;
    %     obj_thrdTrm = beta*trace(invOmega*(mean_c'*mean_c));
    obj_thrdTrm = beta*traceTrm;
    
    obj = 0.5*(obj_frstTrm - obj_scndTrm - obj_thrdTrm);
    
    newbeta_inv = (nytraceTrm + traceTrm)/(n*ny*nx);
    
    newU = newbeta_inv*eye(ny);
    
    % I don't have derivatives here... 
    dobjdU =[];
    
else
    
    %% (1) function value
    
    % invU = diag(abs(10*rand(ny,1)));
%     invU = reshape(invU, ny, ny);
    
    invOmegaKroninvU = kron(invOmega, invU);
    
    obj = logdetns(cov_c*invOmegaKroninvU) - trace(invOmegaKroninvU*cov_c - eye(size(cov_c))) - mean_c'*invOmegaKroninvU*mean_c;
    obj = 0.5*obj;
    
    %% (2) derivative w.r.t. U
    
    % please look at Appendix, in VMstep section for details on derivation
    % eq # is for those in the paper (not in appendix)
    
    [Krm, Knq] =  computeCommutationMatrix(size(invU,1), size(invU,2), size(invOmega,1), size(invOmega,2));
    
    % (2) vec-transposition
    Y = Knq*cov_c;
    P = Knq*(mean_c*mean_c');
    
    Y_n = computeVecTransposition(Y, size(invOmega,2));
    Krm_n = computeVecTransposition(Krm, size(invOmega,2));
    P_n = computeVecTransposition(P, size(invOmega,2));
    
    % Krm_trp_n is K_(n dx dy)^(dy) \trp in eq.(25)
    Krm_trp_n = computeVecTransposition(Krm', size(invOmega,2));
    
    I = speye(size(Krm_n,1)/size(invOmega,2));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % this is where 'out of memory' occurs
    kronIinvOmega = kron(I, invOmega);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    dobjdU_frsTrm = 0.5*n*nx*inv(invU);
    dobjdU_scnTrm = - 0.5*Y_n'*kronIinvOmega*Krm_trp_n;
    dobjdU_trdTrm = - 0.5*P_n'*kronIinvOmega*Krm_trp_n;
    
    dobjdU = dobjdU_frsTrm + dobjdU_scnTrm + dobjdU_trdTrm;
    
    dobjdU = dobjdU(:);
    
    %% update U
    
    newU = 1/(n*nx)*(dobjdU_scnTrm + dobjdU_trdTrm)/(-0.5); % eq.(25)
    
end
