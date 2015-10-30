function [ recorder ] = create_recorder_store_latent( every_iter, only_cov_diag )
%CREATE_RECORDER_STORE_LATENT Construct a recorder which stores all the latent variables in lllvm EM.
%   This recorder does not support multithreading. Only one instance is allowed.
%   See create_recorder().
% Input:
%  - every_iter is to specify how many EM iterations to record once. 
%  This is useful when all variables are big. If unspecified, store every iteration.
%  - If only_cov_diag = true, store only the diagonal of cov_x and cov_c to save 
%  space. Assume only_cov_diag = true if not specified.
%
%@author Wittawat

clear create_recorder_store_latent;
if nargin < 2
    only_cov_diag = true;
end

if nargin < 1
    every_iter = 1;
end

assert(every_iter >= 1);
%recorder = @(st)latent_params(st, every_iter);
if only_cov_diag
    recorder = @latent_params_diag;
else
    recorder = @latent_params;
end

function [rec_v] = latent_params_diag(state )
    % This is a recorder which records all latent parameters. Record only diag 
    % for cov_x and cov_c to save space.
    %
    % Pass nothing to the function to get the recorded variables returned and 
    % clear all the persistent variables used internally.
    %
    persistent rec_vars;

    if nargin < 1
        % No input. return the recorded vars. 
        % Convert all the cells to arrays.
        rec_vars.diag_cov_cs = horzcat(rec_vars.diag_cov_cs{:});
        % each mean_c is dy x n*dx
        %rec_vars.mean_cs = cat(3, rec_vars.mean_cs{:});
        rec_vars.diag_cov_xs = horzcat(rec_vars.diag_cov_xs{:});
        % each mean_x is n*dx x 1
        rec_vars.mean_xs = horzcat(rec_vars.mean_xs{:});
        %
        rec_v = rec_vars;
        % clear
        rec_vars = [];
        return ;
    end
    % start recording 
    if isempty(rec_vars)
        rec_vars = struct();
        % iteration numbers corresponding to the variables recorded.
        rec_vars.i_ems = [];
        rec_vars.alphas = [];
        rec_vars.gammas = [];
        % lower bound values
        rec_vars.lwbs = [];
        rec_vars.diag_cov_cs = {};
        rec_vars.diag_cov_xs = {};
        rec_vars.mean_cs = {};
        rec_vars.mean_xs = {};
    end

    i_em = state.i_em;
    % always store alpha,  gamma, and lwb in every iteration.
    %
    rec_vars.alphas(end+1) = state.alpha;
    rec_vars.gammas(end+1) = state.gamma;
    rec_vars.lwbs(end+1) = state.lwb;
    if mod(i_em-1, every_iter) == 0
        % i_em-1 so that the first iteration is always recorded.
        % record 
        rec_vars.i_ems(end+1) = state.i_em;

        rec_vars.diag_cov_cs{end+1} = diag(state.cov_c);
        rec_vars.diag_cov_xs{end+1} = diag(state.cov_x);
        rec_vars.mean_xs{end+1} = state.mean_x;
        rec_vars.mean_cs{end+1} = state.mean_c;
    end

    rec_v = [];
end

function [rec_v] = latent_params(state )
    % This is a recorder which records all latent parameters.
    %
    % Pass nothing to the function to get the recorded variables returned and 
    % clear all the persistent variables used internally.
    %
    persistent rec_vars;

    %if nargin < 2
    %    % Record every iteration if unspecified.
    %    every_iter = 1;
    %end
    if nargin < 1
        % No input. return the recorded vars. 
        % Convert all the cells to arrays.
        rec_vars.cov_cs = cat(3, rec_vars.cov_cs{:});
        % each mean_c is dy x n*dx
        rec_vars.mean_cs = cat(3, rec_vars.mean_cs{:});
        rec_vars.cov_xs = cat(3, rec_vars.cov_xs{:});
        % each mean_x is n*dx x 1
        rec_vars.mean_xs = horzcat(rec_vars.mean_xs{:});
        %
        rec_v = rec_vars;
        % clear
        rec_vars = [];
        return ;
    end
    % start recording 
    if isempty(rec_vars)
        rec_vars = struct();
        % iteration numbers corresponding to the variables recorded.
        rec_vars.i_ems = [];
        rec_vars.alphas = [];
        rec_vars.gammas = [];
        % lower bound values
        rec_vars.lwbs = [];
        rec_vars.cov_cs = {};
        rec_vars.mean_cs = {};
        rec_vars.cov_xs = {};
        rec_vars.mean_xs = {};
    end

    i_em = state.i_em;
    % always store alpha, gamma, and lwb in every iteration.
    %
    rec_vars.alphas(end+1) = state.alpha;
    rec_vars.gammas(end+1) = state.gamma;
    rec_vars.lwbs(end+1) = state.lwb;

    if mod(i_em-1, every_iter) == 0
        % i_em-1 so that the first iteration is always recorded.
        % record 
        rec_vars.i_ems(end+1) = state.i_em;

        rec_vars.cov_cs{end+1} = state.cov_c;
        rec_vars.mean_cs{end+1} = state.mean_c;
        rec_vars.cov_xs{end+1} = state.cov_x;
        rec_vars.mean_xs{end+1} = state.mean_x;
    end

    rec_v = [];
end

end


