% ibfc_pf_params.m
%  default parameters structure for ibfc_pf
%
% Copyright 2013 Taylor C. Glenn
% tcg@cise.ufl.edu
% see ../LICENSE.txt for license terms

function params = ibfc_pf_params()

params = struct();

params.m = 2;    % fuzzifier
params.beta = 2; % model sparsity weight (bigger is more sparse model)

params.n_particles = 10; % number of particles in the particle filter

params.max_iter = 500; % maximum total iterations

params.converge_thresh = 1e-5; % converged if max loglike changes < this
params.converge_iter = 10;     %  and has stayed like this for > this iterations

params.do_trace = false; % record likelihood at each iteration

params.ll_fun = @ll_ibfc;

end