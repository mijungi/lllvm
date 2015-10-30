function [ ] = compare_testEM_results( s, t )
%COMPARE_TESTEM_RESULTS Compare variables in the two structs returned from testEM
%   - s,t are struct containing all variables returned from running testEM.
%   - s,t are expected to have exactly the same member variables. 
%   - This function checks the discrepancy of the variables in s, t.
%   - Useful for checking the correctness of the code after modifying.
%   - Use s = ws2struct to save all variables in the workspace to s.
%
%@author Wittawat Jitkrittum 
%
%   Example of variables returned from testEM
%% B_qC                       300x1                   4816  double     sparse    
%  C                            3x2x150               7200  double               
%  E_estimates                450x1                   3600  double               
%  G                          150x150               180000  double               
%  H_init                       3x300                 7200  double               
%  H_qX                         3x300                 7200  double               
%  L                          150x150               180000  double               
%  X                            2x150                 2400  double               
%  Y                            3x150                 3600  double               
%  alpha                        1x1                      8  double               
%  alpha_init                   1x1                      8  double               
%  ans                          1x2                     16  double               
%  beta                         1x1                      8  double               
%  beta_init                    1x1                      8  double               
%  cov_c_qX                   300x300              1442408  double     sparse    
%  cov_x_qC                   300x300              1442408  double     sparse    
%  cov_y                      450x450              1620000  double               
%  cov_yest                   450x450              1620000  double               
%  diagU                        1x1                      1  logical              
%  dx                           1x1                      8  double               
%  dy                           1x1                      8  double               
%  epsilon_1                    1x1                      8  double               
%  epsilon_2                    1x1                      8  double               
%  gamma                        1x1                      8  double               
%  gamma_init                   1x1                     32  double     sparse    
%  icount                       1x1                      8  double               
%  invA_qC                    300x300                87528  double     sparse    
%  invA_qC_without_gamma      300x300               720000  double               
%  invGamma_qX                300x300               720000  double               
%  invOmega                   300x300                44968  double     sparse    
%  invPi                      300x300                44968  double     sparse    
%  invPi_init                 300x300                44968  double     sparse    
%  invU                         3x3                     72  double               
%  invU_init                    3x3                     72  double               
%  invV                         3x3                     80  double     sparse    
%  invV_init                    3x3                     80  double     sparse    
%  lwb_C                        1x1                      8  double               
%  lwb_likelihood               1x1                      8  double               
%  lwb_x                        1x1                      8  double               
%  max_em_iter                  1x1                      8  double               
%  mean_c_qX                    3x300                 7200  double               
%  mean_x_qC                  300x1                   4816  double     sparse    
%  mu_y                       450x1                   3600  double               
%  mu_yest                    450x1                   3600  double               
%  n                            1x1                      8  double               
%  newAlpha                     1x1                      8  double               
%  newGamma                     1x1                      8  double               
%  newU                         3x3                     72  double               
%  oldRng                       1x1                   3046  struct               
%  prec_Yest                  450x450              1620000  double               
%  seed                         1x1                      8  double               
%  thresh_lwb                   1x1                      8  double               
%  variational_lwb             20x1                    160  double               
%  vc                         900x1                   7200  double               
%  vx                         300x1                   2400  double               
%  vy                         450x1                   3600  double               
%
%

assert(isstruct(s));
assert(isstruct(t));
abs_tol = 1e-5;
%fName = fieldnames(s);
fName = {'B_qC', 'B_qC', 'C', 'E_estimates', 'G',  'H_qX', 'L', 'X', 'Y', 'alpha', 'alpha_init', 'beta', 'beta_init', 'cov_c_qX', 'cov_x_qC', 'cov_y', 'cov_yest', 'diagU', 'dx', 'dy', 'epsilon_1', 'epsilon_2', 'gamma', 'gamma_init', 'invA_qC',  'invGamma_qX', 'invOmega',  'invPi_init', 'invU', 'invU_init', 'invV', 'invV_init', 'lwb_C', 'lwb_likelihood', 'lwb_x', 'max_em_iter', 'mean_c_qX', 'mean_x_qC', 'mu_y', 'mu_yest', 'n', 'newAlpha', 'newGamma', 'newU',  'thresh_lwb', 'variational_lwb', 'vc', 'vx', 'vy'};
for i=1:length(fName)
    f = fName{i};
    svar = s.(f);
    tvar = t.(f);
    if any(abs(svar(:) - tvar(:)) >= abs_tol)
        display(sprintf('Entries of %s differ.', f));
    end
end

%if any(abs(s.C(:) - t.C(:)) >= abs_tol)
%    display(sprintf('Entries of C differ.'));
%end

%if any(abs(s.cov_c_qX(:) - t.cov_c_qX(:)) >= abs_tol)
%    display(sprintf('Entries of cov_c_qX differ.'));
%end

%if any(abs(s.cov_x_qC(:) - t.cov_x_qC(:)) >= abs_tol)
%    display(sprintf('Entries of cov_x_qC differ.'));
%end

%if any(abs(s.mean_c_qX(:) - t.mean_c_qX(:)) >= abs_tol)
%    display(sprintf('Entries of mean_c_qX differ.'));
%end

%if any(abs(s.mean_x_qC(:) - t.mean_x_qC(:)) >= abs_tol)
%    display(sprintf('Entries of mean_x_qC differ.'));
%end


end

