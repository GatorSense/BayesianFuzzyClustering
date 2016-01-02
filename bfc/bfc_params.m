% bfc_params.m
%  default parameters structure for bfc sampler
%
% Copyright 2013 Taylor C. Glenn
% tcg@cise.ufl.edu
% see ../LICENSE.txt for license terms

function p = bfc_params

% parameters for Bayesian Fuzzy Clustering sampler

p = struct();

p.n_comp = 2;
p.m = 2;
p.n_iter = 1000;
p.alpha = 1;

p.mem_prop_alpha = 1; % dirichlet parameter for membership proposal distribution

p.rpt_ival = 50; % reporting interval

p.re_seed_rng = false; % reset random number generator to fixed seed to get same answer every run

p.fcm_init = false; % if true, run fcm for a few iterations with m=2 to init centers and memberships

p.do_trace = false; % record likelihood at each iteration

p.figno = 12; % output figure number

end