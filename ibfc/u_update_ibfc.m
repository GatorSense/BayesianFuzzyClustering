% u_update_ibfc.m
%  ibfc membership update function
%
% Copyright 2013 Taylor C. Glenn
% tcg@cise.ufl.edu
% see ../LICENSE.txt for license terms

function U = u_update_ibfc(X,C,p)

K = size(C,1);
N = p.N;

U = zeros(N,K);

idsq = zeros(N,K);
for i=1:K
    W = X - repmat(C(i,:),N,1);
    idsq(:,i) = 1./sum(W.*W,2);
end
sum_idsq = sum(idsq,2);

for i=1:K    
    U(:,i) = (idsq(:,i)./sum_idsq).^(1/(p.m-1));
end

end