classdef GammaUtils
    %GAMMAUTILS A list of static functions for computing <Gamma>
    %@author Wittawat
    
    properties
    end

    methods(Static)

        function [ Gam ] = compute_Gamma_ultimate(Ltilde, spLogG, EXX, dx)
            % An improved version of compute_Gamma_n3. Memory: O(n^2)
            %This is the fastest function .
            % - Ltilde: n x n 
            % - spLogG: G in logical (not double).  Can be sparse 
            % - EXX: dx*n x dx*n matrix where each dx x dx block is E[xx^T]
            % - dx: an integer. dimensionality of x.
            %

            n = size(Ltilde, 1);
            AllCoeff = zeros(n, n, dx, dx);
            for r=1:dx 
                for s=1:dx
                    EXX_rs = EXX(r:dx:end, s:dx:end);
                    Coeff_rs = GammaUtils.compute_coeff_Gamma_nofactor(Ltilde, spLogG, EXX_rs);
                    AllCoeff(:, :, r, s) = Coeff_rs;
                end
            end
            AllCoeff = permute(AllCoeff, [3 4 1 2]);
            % See http://www.ee.columbia.edu/~marios/matlab/Matlab%20array%20manipulation%20tips%20and%20tricks.pdf
            % section 6.1.1 to understand what I am doing.
            AllCoeff = permute(AllCoeff, [1 3 2 4]);
            Gam = reshape(AllCoeff, [dx*n, dx*n]);

        end % end compute_Gamma_n3

        function Coeff = compute_coeff_Gamma_nofactor(Ltilde, spLogG, C)
            % An improved version of compute_coeff_Gamma. Memory 
            % requirement: O(n^2)
            %  - C is square but not necessarily symmetric. 
            %

            UTV = C;
            G = double(spLogG);
            B = G + diag(sum(G, 1));

            UTVG = UTV*G;
            LTLG = Ltilde*G;
            GTUTV = G'*UTV;
            GTL2 = G'*Ltilde;

            %line1 = B'*(Ltilde.*UTV)*B - B'*(Ltilde.*UTVG) - B'*(LTLG.*UTV);
            line1 = B'*( (Ltilde.*UTV)*B - (Ltilde.*UTVG) - (LTLG.*UTV) );
            line2 = -(Ltilde.*GTUTV)*B + Ltilde.*(G'*UTVG) + GTUTV.*LTLG;
            line3 = -(GTL2.*(UTV))*B  +GTL2.*UTVG + UTV.*(G'*LTLG);

            Coeff = line1 + line2 + line3;
        end

        function [ Gam ] = compute_Gamma_n3(Ltilde, spLogG, EXX, dx)
            %COMPUTE_GAMMA_N3 Compute <Gamma> using O(n^3) memory. 
            %Wittawat: This function is deprecated. Use compute_Gamma_ultimate.
            % - Ltilde: n x n 
            % - spLogG: G in logical (not double).  Can be sparse 
            % - EXX: dx*n x dx*n matrix where each dx x dx block is E[xx^T]
            % - dx: an integer. dimensionality of x.
            %

            % decompose Ltilde into La'*La
            [U, V] = eig(Ltilde);
            La = diag(sqrt(diag(V)))*U';
            n = size(La, 1);
            AllCoeff = zeros(n, n, dx, dx);
            for r=1:dx 
                for s=1:dx
                    % TODO: We might be able to do SVD on ECTC once and pick the rows and columns 
                    % of the SVD factors instead of doing SVD for each (r,s).

                    % n x n covariance matrix. Each element i,j is E[(x_i^\top x_j)_{r,s}]
                    % EXX_rs may not be positive definite if r != s.
                    EXX_rs = EXX(r:dx:end, s:dx:end);
                    if r==s
                      % decompose M into T'*T
                      [UC, VC] = eig(EXX_rs);
                      T = diag(sqrt(diag(VC)))*UC';
                      Ubar = GammaUtils.Gamma_factor(La, T, spLogG);
                      Coeff_rs = (Ubar'*Ubar);
                    else 
                       Coeff_rs = GammaUtils.compute_coeff_Gamma(La, spLogG, EXX_rs, dx);
                    end
                    AllCoeff(:, :, r, s) = Coeff_rs;
                end
            end
            AllCoeff = permute(AllCoeff, [3 4 1 2]);
            % See http://www.ee.columbia.edu/~marios/matlab/Matlab%20array%20manipulation%20tips%20and%20tricks.pdf
            % section 6.1.1 to understand what I am doing.
            AllCoeff = permute(AllCoeff, [1 3 2 4]);
            Gam = real(reshape(AllCoeff, [dx*n, dx*n]));

        end % end compute_Gamma_n3

        function Coeff = compute_coeff_Gamma(La, spLogG, C, dx)
            % Much like compute_Gam_scaled_iden. However return the actual coefficients 
            % before taking a Kronecker product with I_dx.
            %  - La: A factor such that La'*La = Ltilde.
            %  - C is square but not necessarily symmetric. 
            %

            % decompose CCscale = U'*V 
            [u,s,v] = svd(C);
            U = diag(sqrt(diag(s)))*u';
            V = diag(sqrt(diag(s)))*v';

            Ubar = GammaUtils.Gamma_factor(La, U, spLogG);
            Vbar = GammaUtils.Gamma_factor(La, V, spLogG);
            Coeff = (Ubar'*Vbar);
        end


        function M = Gamma_factor(La, U, spLogG )
            % Return a matrix M : n*size(U, 1) x n

            % TODO: The following code requires O(n^3) storage requirement. 
            n = size(La, 1);
            nu = size(U, 1);
            Gd = double(spLogG);
            Degs = sum(Gd, 1);
            LaU = reshape(MatUtils.colOuterProduct(La, U), [n*nu, n]);
            %M2 = LaU*Gd;
            %M2 = M2 +bsxfun(@times, LaU, Degs);
            M2 = LaU*(Gd+diag(Degs));
            M2 = M2 -reshape(MatUtils.colOuterProduct(La, U*Gd), [n*nu, n]);
            M2 = M2 -reshape(MatUtils.colOuterProduct(La*Gd, U), [n*nu, n]);
            M = M2;

            %n = size(La, 1);
            %M = zeros(n*n, size(U, 1));
            %for i=1:n
            %  Gi = spLogG(i, :);
            %  Lai = La(:, i);
            %  Ui = U(:, i);

            %  s1 = La(:, Gi)*U(:, Gi)';
            %  s2 = sum(Gi)*Lai*Ui';
            %  s3 = -Lai*sum(U(:, Gi), 2)';
            %  s4 = -sum(La(:, Gi), 2)*Ui';

            %  M(:, i) = reshape(s1+s2+s3+s4, [n*size(U, 1), 1]);
            %end

            %display(sprintf('Gamma_factor: |M-M2| = %.6f', norm(M-M2, 'fro') ));
        end

    end % end methods

end

