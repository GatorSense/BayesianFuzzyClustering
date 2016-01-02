% test.m
% test script for ibfc, 4 clusters
%   two large spheres and two small spheres
%
% Copyright 2013 Taylor C. Glenn
% tcg@cise.ufl.edu
% see ../LICENSE.txt for license terms


RandStream.setDefaultStream(RandStream('mt19937ar','Seed', 5489));

m = 2;
% generate some data

n_dim = 2;
n_pts_per = 200;
n_comp = 4;

mu1 = [2; 2];
mu2 = [2; 9];
mu3 = [9; 2];
mu4 = [9; 9];

c1 = 2*[1 0;0 1];
c2 = .25*[1 0;0 1];
c3 = .75*[1 0;0 1];
c4 = 2*[1 0;0 1];


x1 = chol(c1)*randn(n_dim,n_pts_per) + repmat(mu1,1,n_pts_per);
x2 = chol(c2)*randn(n_dim,n_pts_per) + repmat(mu2,1,n_pts_per);
x3 = chol(c3)*randn(n_dim,n_pts_per) + repmat(mu3,1,n_pts_per);
x4 = chol(c4)*randn(n_dim,n_pts_per) + repmat(mu4,1,n_pts_per);

X = [x1 x2 x3 x4];

%--- fcm C=4
[C,U] = fcm(X',n_comp);

[~,label] = max(U,[],1);

figure(11);
clf;
scatter(X(1,:),X(2,:),[],label);
hold on;
for i=1:n_comp
    text(C(i,1),C(i,2),num2str(i),'BackgroundColor','w','Color','k');
end
title('FCM');
xlabel('D1');
ylabel('D2');
axis equal;


%--- ibfc
pr = ibfc_pf_params();
pr.m = m;
pr.n_particles = 10;
pr.max_iter = 50;
pr.converge_iter = 50;
pr.alpha = 2;
[iC,iU,iK,ill,mx] = ibfc_pf(X',pr);

%---- display max likelihood cluster output
[~,l_iU] = max(iU,[],2);
figure(12);
clf;
scatter(X(1,:),X(2,:),[],l_iU');
hold on;
for i=1:size(iC,1)
    text(iC(i,1),iC(i,2),num2str(i),'BackgroundColor','w','Color','k');
end
title({'IBFC',sprintf('K=%d LogLikelihood=%.3f',iK,ill)});
xlabel('D1');
ylabel('D2');
axis equal;


%---- display other, less likely, cluster outputs
sizes = [2 3 5 6];
for i=1:numel(sizes)
    
    s = mx(sizes(i));
    [~,l_s] = max(s.U,[],2);
    
    figure(20+i);
    clf;
    scatter(X(1,:),X(2,:),[],l_s');
    hold on;
    for k=1:size(s.C,1)
        text(s.C(k,1),s.C(k,2),num2str(k),'BackgroundColor','w','Color','k');
    end
    title({'IBFC',sprintf('K=%d LogLikelihood=%.3f',s.K,s.ll)});
    xlabel('D1');
    ylabel('D2');
    axis equal;
        
end




