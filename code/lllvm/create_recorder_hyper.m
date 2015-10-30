function [ recorder ] = create_recorder_hyper()
%CREATE_RECORDER_HYPER Construct a recorder which stores only the hyperparameters.
%   - This recorder does not support multithreading. Only one instance is allowed.
%   - Hyperparameters include alpha,  gamma. Also lower bound values in 
%   every iteration.
%
%@author Wittawat

clear create_recorder_hyper;

recorder = @record_hyper;

function [rec_v] = record_hyper(state )
    % Pass nothing to the function to get the recorded variables returned and 
    % clear all the persistent variables used internally.
    %
    persistent rec_vars;

    if nargin < 1
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
    end

    % always store alpha,  gamma, and lwb in every iteration.
    %
    rec_vars.i_ems(end+1) = state.i_em;
    rec_vars.alphas(end+1) = state.alpha;
    rec_vars.gammas(end+1) = state.gamma;
    rec_vars.lwbs(end+1) = state.lwb;

    rec_v = [];
end

end


