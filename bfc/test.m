% test.m
%  test script comparing FCM and Bayesian Fuzzy Clustering Sampler
%
% Copyright 2013 Taylor C. Glenn
% tcg@cise.ufl.edu
% see ../LICENSE.txt for license terms

RandStream.setDefaultStream(RandStream('mt19937ar','Seed', 5489));

% generate some data
n_dim = 2;
n_pts = 500;
n_comp = 2;

c1 = [1 0;0 1];
mu1 = [1; 2];

c2 = [1 0; 0 1];
mu2 = [4; 4];

x1 = chol(c1)*randn(n_dim,n_pts/2) + repmat(mu1,1,n_pts/2);
x2 = chol(c2)*randn(n_dim,n_pts/2) + repmat(mu2,1,n_pts/2);

X = [x1 x2];

figure(10);
clf;
scatter(x1(1,:),x1(2,:),'bo');
hold on;
scatter(x2(1,:),x2(2,:),'rs');

%--- fcm m=2
m = 2;
[C,U] = fcm(X',2,[m 100 1e-5 1]);

figure(11);
clf;
subplot(2,1,1);
scatter(X(1,:),X(2,:),[],U(1,:)');
hold on;
text(C(1,1),C(1,2),'1','BackgroundColor','w','Color','k');
text(C(2,1),C(2,2),'2','BackgroundColor','w','Color','k');
title({'FCM m=2','membership in cluster 1'});
xlabel('D1');
ylabel('D2');
axis equal;

subplot(2,1,2);
plot(U(1,:),'r-');
hold on;
%plot(U(2,:),'b-');
xlabel('index');
ylabel('membership');
ylim([-0.01,1.01]);


%--- fcm m=10
m = 10;
[C,U] = fcm(X',2,[m 100 1e-5 1]);


figure(13);
clf;
subplot(2,1,1);
scatter(X(1,:),X(2,:),[],U(1,:)');
hold on;
text(C(1,1),C(1,2),'1','BackgroundColor','w','Color','k');
text(C(2,1),C(2,2),'2','BackgroundColor','w','Color','k');
title({'FCM m=10','membership in cluster 1'});
xlabel('D1');
ylabel('D2');
axis equal;

subplot(2,1,2);
plot(U(1,:),'r-');
hold on;
%plot(U(2,:),'b-');
xlabel('index');
ylabel('membership');
ylim([-0.01,1.01]);


%--- bfc m=2
p1 = bfc_params;
p1.re_seed_rng = true;
p1.n_comp = 2;
p1.m = 2;
p1.n_iter = 1000;
p1.alpha = 1;
p1.fcm_init = false;

[tC,tU] = bfc_sampler(X',p1);


%--- bfc m=10
p2 = p1;
p2.m = 10;
p2.figno = 14;
[tC,tU] = bfc_sampler(X',p2);

%--- bfc m=1
p3 = p1;
p3.m = 1;
p3.figno = 15;
[tC,tU] = bfc_sampler(X',p3);
subplot(2,1,2);
ylim([-0.1 1.1]);

%--- bfc m=-10
p4 = p1;
p4.m = -10;
p4.figno = 16;
[tC,tU] = bfc_sampler(X',p4);

%--- bfc m=2 alpha=3
p5 = p1;
p5.alpha = 3;
p5.figno = 17;
[tC,tU] = bfc_sampler(X',p5);

%--- bfc m=4 alpha=.95
p6 = p1;
p6.m = 4;
p6.alpha = 0.95;
p6.figno = 18;
[tC,tU] = bfc_sampler(X',p6);




