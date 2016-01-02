% u_update_ibfc.m
%  ibfc cluster prototype update function
%
% Copyright 2013 Taylor C. Glenn
% tcg@cise.ufl.edu
% see ../LICENSE.txt for license terms

function C = c_update_ibfc(X,U,p)

K = size(U,2);
D = p.D;

C = zeros(K,D);

Um = U.^p.m;

sumUm = sum(Um,1);
eD = eye(D);
for k=1:K
    xbar = sum(bsxfun(@times,X,Um(:,k)),1);
    C(k,:) = (xbar+p.mu_c*p.siginv_c)/(sumUm(k)*eD+p.siginv_c);
end

end