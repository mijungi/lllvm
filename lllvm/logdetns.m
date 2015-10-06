function [x ] = logdetns(A)
   
    [x ] = logdetns_wittawat(A);
    %[x, traceA] = logdetns_old(A);
end

% ----------------------------

function [x ] = logdetns_wittawat(A)
    %
    %http://mathoverflow.net/questions/43436/finding-the-determinant-of-a-matrix-with-lu-composition
    %http://math.stackexchange.com/questions/831823/finding-determinant-of-44-matrix-via-lu-decomposition
    % Wittawat: I really want a proper reference on LU and log det.
    
    if checksym(A)  % crude symmetry check
       x = 2*sum(log(diag(chol(A))));
    else
       %disp('this is not a symm pd mat');
       x = sum(log(abs(diag(lu(A)))));
    end
    %eigv = eig(A);
    %x = sum(log(real(eigv)));

    %if any( abs(imag(eigv) ) > 1e-3)
    %    %warning('logdetns: eigenvalues have large imaginary components.');
    %end
    %if any(real(eigv) < -1e-3)
    %    %warning('logdetns: eigenvalues are large negative reals.');
    %end
    %display(sprintf('log det: %.4f', x ));
end

function z = checksym(A)
% Crude (but fast) algorithm for checking symmetry, just by looking at the
% first column and row

%z = isequal(A(:,1),A(1,:)');

n = size(A, 1);
m = min(6, n);
Asub = A(1:m, 1:m);
z = norm(Asub - Asub', 'fro') < 1e-6;

end 

function [x, traceA] = logdetns_old(A)

    % LOGDET - computes the log-determinant of a matrix A using Cholesky or LU
    % factorization
    %
    % LOGDET
    %
    % x = logdet(A);
    %
    % This is faster and more stable than using log(det(A))
    %
    % Input:
    %     A NxN - A must be sqaure, positive SYMMETRIC and semi-definite
    %     (Chol assumes A is symmetric)

    if checksym(A)  % crude symmetry check

        % Wittawat: This condition makes no sense. If det can be computed, 
        % why not just return its log ? Why bother with chol, svd ?
        if det(A)~=0 
            x = 2*sum(log(diag(chol(A))));
            traceA = []; 
        else
            [u, d, v] = svd(A);
            x = sum(log(diag(d)));
            traceA = sum(diag(d)); 
        end

    else
        %disp('this is not a symm pd mat');

        x = sum(log(abs(diag(lu(A)))));
        traceA = [];
    end
end
