% ibfc_pf.m
% Infinite Bayesian Fuzzy Clustering Particle Filter
%
% Copyright 2013 Taylor C. Glenn
% tcg@cise.ufl.edu
% see ../LICENSE.txt for license terms

function [C,U,K,ll,mx,trace] = ibfc_pf(X,params)

% ibfc, infinite bayesian fuzzy clustering
%  a particle filter based search for MAP values of C,U,K

P = params.n_particles;
m = params.m;
alpha = params.beta; % model sparsity weight (called beta in paper, alpha here)
n_iter = params.max_iter;

[N,D] = size(X);

p = struct(); %parameters
p.N = N;
p.D = D;
p.m = m;
p.mu_x = mean(X,1);
p.sigma_c = 3*cov(X);
p.rt_sigma_c = chol(p.sigma_c);
p.siginv_c = pinv(p.sigma_c);
p.logdet_sigma_c = logdet(p.sigma_c);
p.mu_c = p.mu_x;
p.C1 = -D/2*log(2*pi); % common gaussian constant
p.alpha = alpha;

%P = 5; % number of sample particles to use

ll_fun = params.ll_fun;

% initialize
s = struct();
s.K = 1;
s.C = randn(1,D)*p.rt_sigma_c + p.mu_c;
s.U = u_update_ibfc(X,s.C,p);
s.ll = ll_fun(X,s,p);

for i=1:P
    pcl(i) = s;
end

mx = s;

if params.do_trace
   trace = nan(1,n_iter+1);
   trace(1) = mx.ll;
else
    trace = [];
end

n_update = 1;
mxll = -inf;
converge_ctr = 0;

for iter=1:n_iter

    for i=1:P
        s = pcl(i);
        K = poissrnd(s.K);
        if K < s.K
            rp = randperm(s.K);
            inds = rp(1:K);
            pcl(i).K = K;
            pcl(i).C = s.C(inds,:);
           
        elseif K > s.K
            n_new = K-s.K;
            new_C = randn(n_new,D)*p.rt_sigma_c + repmat(p.mu_c,n_new,1);
            pcl(i).K = K;
            pcl(i).C = [s.C; new_C];
        else
            pcl(i).K = s.K;
            pcl(i).C = s.C;
        end
       
        for update=1:n_update
            %update Us of each particle
            pcl(i).U = u_update_ibfc(X,pcl(i).C,p);
           
            %update C for each particle
            pcl(i).C = c_update_ibfc(X,pcl(i).U,p);
        end
        %compute likelihood of particle
        pcl(i).ll = ll_fun(X,pcl(i),p);        
        mx = update_mx(pcl(i),mx);
    end
        
    %sample among particles by importance
    not_empty_mx = ~cellfun(@isempty,{mx.ll});
    pcls = [pcl mx(not_empty_mx)];    
    lls = [pcls.ll];
    prev_mxll = mxll;
    mxll = max(lls);
    
    probs = exp(lls - (log(sum(exp(lls - mxll)))+mxll));
    inds = sample(probs,P);
    
    pcl = pcls(inds);
                
    %------ reporting ---------    
    if mod(iter,50) == 0
                
        fprintf('------\n');
        for sz = 1:numel(mx)
            fprintf('sz %d: %f\n',sz,mx(sz).ll);
        end
        fprintf('------\n');                
    end
        
    if mod(iter,10) == 0
        mx_lls = -inf(1,numel(mx));
        for sz = 1:numel(mx)
            if ~isempty(mx(sz).ll), mx_lls(sz) = mx(sz).ll; end
        end
        [mx_ll,mx_ind] = max(mx_lls);
        fprintf('%d: max ll: %f (%d), curr: ',iter,mx_ll,mx_ind);
        fprintf('%d ',[pcl.K]);
        fprintf('\n');
    end
    
    if params.do_trace
        trace(iter) = mxll;
    end
    
    %------ check for convergence -----
    change = mxll-prev_mxll;
    if change < params.converge_thresh
        converge_ctr = converge_ctr+1;
        if converge_ctr >= params.converge_iter
            fprintf('Converged\n');
            break;
        end
    else
        converge_ctr = 0;
    end
end

mx_lls = -inf(1,numel(mx));     
for sz = 1:numel(mx)
    if ~isempty(mx(sz).ll), mx_lls(sz) = mx(sz).ll; end
end       
[ll,mx_ind] = max(mx_lls);

C = mx(mx_ind).C;
U = mx(mx_ind).U;
K = mx(mx_ind).K;

end

function mx = update_mx(s,mx)

if s.K > 0 && (s.K > numel(mx) || isempty(mx(s.K).ll) || s.ll > mx(s.K).ll)
    mx(s.K) = s;
end

end