% ll_ibfc.m
%  ibfc log-likelihood evaluation function(s)
%
% Copyright 2013 Taylor C. Glenn
% tcg@cise.ufl.edu
% see ../LICENSE.txt for license terms

function ll = ll_ibfc(X,s,p)
% full (joint) likelihood of data X and sample s given parameters p
if isempty(s.U)
    ll = -inf;
    return;
end
llx = ll_X_given_UCK(X,s,p);
llu = ll_U(s,p);
llc = ll_C(s,p);
llk = ll_K(s,p);

ll = llx+llu+llc+llk;

end

function ll = ll_X_given_UCK(X,s,p)
% data likelihood

N = p.N;
K = size(s.C,1);

Um = s.U.^p.m;
WW = zeros(N,K);
for k=1:K
    W = X - repmat(s.C(k,:),N,1);
    WW(:,k) = sum(W.*W,2);
end

ll = N*p.C1 - .5/K*sum(sum(Um.*WW));  %average over K

end

function ll = ll_U(s,p)
%memberships

K = size(s.U,2);

ll = p.alpha*p.N/(K);

end

function llc = ll_C(s,p)
% cluster centers
K = size(s.C,1);

W = s.C - repmat(p.mu_c,K,1);
WW = sum((W*p.siginv_c).*W,2);

llc = sum(p.C1 - p.logdet_sigma_c - .5*WW)/K;

end

function llk = ll_K(s,p)
% number of clusters

% K ~ poisson(alpha * log(N))
%  likelihood(k|lambda) = lambda^k/k! * exp(-lambda)
%  ll = k*log(lambda) - sum(log(1:k)) - lambda

lambda = 1 * log(p.N); %fixme: maybe put a p.beta parameter to weight lambda
llk = s.K* log(lambda) - sum(log(1:s.K)) - lambda;

end


