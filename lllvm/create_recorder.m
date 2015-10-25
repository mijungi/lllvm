function [recorder ] = create_recorder(rec_type )
%GEN_RECORDER create a recorder to be used with lllvm.m.
%   A recorder is a function taking in all variables in an EM iteration and 
%   does something.
%
%   Input:
%   - rec_type: a string denoting a recorder type. See the code.
%
% @author Wittawat 21 May 2015
%

if strcmp(rec_type, 'print_struct')
    recorder = @rec_print_struct;
else
    error('unknown recorder type');
end

end


function rec_print_struct(state)
% rec_print_struct is a simple recorder which prints all relevant variables in 
% each EM iteration.
%
    toprint = struct();
    toprint.i_em = state.i_em;
    toprint.alpha = state.alpha;
    toprint.gamma = state.gamma;
    disp(toprint);

end

