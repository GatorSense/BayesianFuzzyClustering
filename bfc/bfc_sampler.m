% bfc_sampler.m
%  Bayesian Fuzzy Clustering MCMC sampler
%
% Copyright 2013 Taylor C. Glenn
% tcg@cise.ufl.edu
% see ../LICENSE.txt for license terms

function [C,U,ll,trace] = bfc_sampler(X,params)

n_comp = params.n_comp;
m = params.m;
n_iter = params.n_iter;
alpha = params.alpha;

if params.re_seed_rng
    RandStream.setDefaultStream(RandStream('mt19937ar','Seed', 5489));
    randomseed([101,1001,10001]); % set random seed for lightspeed library 
end

[n_pts,n_dim] = size(X);

% proposal parameters
sigma_prop = .25*cov(X);
sqrt_sigma_prop = chol(sigma_prop);

% hyperparameters
mu_c = mean(X,1);
sigma_c = 3*cov(X);
siginv_c = pinv(sigma_c);

% initialize centers, memberships
if params.fcm_init
    [C,Ut] = fcm(X,n_comp,[2,20,1e-5,1]);
    U = Ut';
else    
    rp = randperm(n_pts);
    C = X(rp(1:n_comp),:);
    
    D = zeros(n_pts,n_comp);
    U = zeros(n_pts,n_comp);
    for i=1:n_comp
        D(:,i) = 1./(sum((X - repmat(C(i,:),n_pts,1)).^2, 2)+eps);
    end
    sumD = sum(D,2);
    for i=1:n_comp
        U(:,i) = D(:,i)./sumD;
    end
end


drawfig(X,C,U,m,alpha,params.figno);

C_max = C;
U_max = U;


ll_fun = @(X,C,U,m,mu_c,siginv_c)(fcm_ll_dirichlet(X,C,U,m,mu_c,siginv_c,alpha));
u_ll_fun = @(X,C,U,m)(u_lls_dirichlet(X,C,U,m,alpha));

% ll_fun = @(X,C,U,m,mu_c,siginv_c)(fcm_ll_invgamma(X,C,U,m,mu_c,siginv_c,alpha));
% u_ll_fun = @(X,C,U,m)(u_lls_invgamma(X,C,U,m,alpha));

% ll_fun = @fcm_ll_noprior;
% u_ll_fun = @u_lls_noprior;

%ll_fun = @fcm_ll;
%u_ll_fun = @u_lls;

ll = ll_fun(X,C,U,m,mu_c,siginv_c);
ll_max = ll;
ll_max_display = ll_max;

if params.do_trace
    trace = zeros(1,n_iter+1);
    trace(1) = ll;
else
    trace = [];
end

u_accepted = 0;
c_accepted = 0;

for iter=1:n_iter

    %generate new U's
    % use uniform proposal
    U_samp = dirichlet_sample(params.mem_prop_alpha*ones(1,n_comp),n_pts);
    
    lls_old = u_ll_fun(X,C,U,m);    
    lls_proposed = u_ll_fun(X,C,U_samp,m);
    
    % decide to accept each u vector or not
    accept_U = exp(lls_proposed-lls_old) > rand(n_pts,1);
    U(accept_U,:) = U_samp(accept_U,:);
    
    u_accepted = u_accepted + sum(accept_U);
    
    % check each sample for inclusion in the max likelihood set
    lls_new_max_c = u_ll_fun(X,C_max,U_samp,m);
    lls_old_max_c = u_ll_fun(X,C_max,U_max,m);
    accept = lls_new_max_c > lls_old_max_c;
    U_max(accept,:) = U_samp(accept,:);
        
    %generate new C's
    C_samp = randn(n_comp,n_dim)*sqrt_sigma_prop + C;
    new_C = C;
    new_C_max = C_max;    
    for i=1:n_comp
        for d=1:n_dim
            %test each sample center for acceptance
            old_ll = ll_fun(X,new_C,U,m,mu_c,siginv_c);
            
            new_C(i,d) = C_samp(i,d);
            
            new_ll = ll_fun(X,new_C,U,m,mu_c,siginv_c);
            
            if (exp(new_ll-old_ll) > rand(1))                
                c_accepted = c_accepted + 1;
            else
                new_C(i,d) = C(i,d);
            end
            
            %check sample for inclusion in max set
            new_C_max(i,d) = C_samp(i,d);
            
            ll_new_c = ll_fun(X,new_C_max,U_max,m,mu_c,siginv_c);
            ll_old_c = ll_fun(X,C_max,U_max,m,mu_c,siginv_c);
            if ll_new_c > ll_old_c
                C_max(i,d) = C_samp(i,d);
            else
                new_C_max(i,d) = C_max(i,d);
            end
        end
    end
    C = new_C;
    
    % check full sample for new max
    ll_full = ll_fun(X,C,U,m,mu_c,siginv_c);
    ll_max  = ll_fun(X,C_max,U_max,m,mu_c,siginv_c);
    if ll_full > ll_max
       ll_max = ll_full;
       C_max = C;
       U_max = U;
    end
        
    %fprintf('.');    
    if mod(iter,params.rpt_ival) == 0
        u_rate = u_accepted/(params.rpt_ival*n_pts);
        c_rate = c_accepted/(params.rpt_ival*n_comp*n_dim);
        u_accepted = 0;
        c_accepted = 0;
            
        if ll_max > ll_max_display
            ll_max_display = ll_max;
            drawfig(X,C_max,U_max,m,alpha,params.figno);
        end
        %drawfig(X,C,U);
        
        fprintf('iter %d:  U rate: %.3f  C rate: %.3f  max: %f\n',iter,u_rate,c_rate,ll_max);        
    end
    
    if params.do_trace
        trace(iter+1) = ll_max;
    end
end

C = C_max;
U = U_max';
ll = ll_max;

end

function ll = fcm_ll(X,C,U,m,mu_c,siginv_c)

  [n_pts,n_dim] = size(X);
  n_comp = size(C,1);
  
  %sum of gaussian lls with u_ij precision
  ll = 0;
      
  % sum of exponent terms
  Z = zeros(n_pts,n_comp);
  for i=1:n_comp
      Z(:,i) = sum((X - repmat(C(i,:),n_pts,1)).^2, 2);
  end
  ll = ll + -.5* sum((U(:).^m).*Z(:));

  % with fuzzy membership prior, base term of normal and
  %  prior likelihoods cancel

  %sum of C lls in guassian with mu_c,siginv_c
  for i=1:n_comp
      z = C(i,:)-mu_c;
      ll = ll + -.5* z*siginv_c*z';
  end
  
end

function ll = fcm_ll_invx2(X,C,U,m,mu_c,siginv_c)

  [n_pts,n_dim] = size(X);
  n_comp = size(C,1);
  
  %sum of gaussian lls with u_ij precision
  ll = 0;
      
  % sum of exponent terms
  Z = zeros(n_pts,n_comp);
  for i=1:n_comp
      Z(:,i) = sum((X - repmat(C(i,:),n_pts,1)).^2, 2);
  end
  ll = ll + -.5* sum((U(:).^m).*Z(:));

  % with fuzzy membership prior, base term of normal and
  %  first term of inv chi-sq prior cancel, leaves exponent term
  ll = ll - 1/2*sum(sum(1./U));

  %sum of C lls in guassian with mu_c,siginv_c
  for i=1:n_comp
      z = C(i,:)-mu_c;
      ll = ll + -.5* z*siginv_c*z';
  end
  
end

function ll = fcm_ll_invgamma(X,C,U,m,mu_c,siginv_c,beta)

  [n_pts,n_dim] = size(X);
  n_comp = size(C,1);
  
  %sum of gaussian lls with u_ij precision
  ll = 0;
      
  % sum of exponent terms
  Z = zeros(n_pts,n_comp);
  for i=1:n_comp
      Z(:,i) = sum((X - repmat(C(i,:),n_pts,1)).^2, 2);
  end
  ll = ll + -.5* sum((U(:).^m).*Z(:));

  % with fuzzy membership prior, base term of normal and
  %  first term of inv chi-sq prior cancel, leaves exponent term
  %beta = 0.00001;
  ll = ll - beta*sum(sum(1./U));

  %sum of C lls in guassian with mu_c,siginv_c
  for i=1:n_comp
      z = C(i,:)-mu_c;
      ll = ll + -.5* z*siginv_c*z';
  end
  
end

function ll = fcm_ll_noprior(X,C,U,m,mu_c,siginv_c)

  [n_pts,n_dim] = size(X);
  n_comp = size(C,1);
  
  %sum of gaussian lls with u_ij precision
  ll = 0;
      
  % sum of exponent terms
  Z = zeros(n_pts,n_comp);
  for i=1:n_comp
      Z(:,i) = sum((X - repmat(C(i,:),n_pts,1)).^2, 2);
  end
  ll = ll + m*n_dim/2*sum(log(U(:))) -.5* sum((U(:).^m).*Z(:));


  %sum of C lls in guassian with mu_c,siginv_c
  for i=1:n_comp
      z = C(i,:)-mu_c;
      ll = ll + -.5* z*siginv_c*z';
  end
  
end

function ll = fcm_ll_dirichlet(X,C,U,m,mu_c,siginv_c,alpha)

  [n_pts,n_dim] = size(X);
  n_comp = size(C,1);
  
  %sum of gaussian lls with u_ij precision
  ll = 0;
      
  % sum of exponent terms
  Z = zeros(n_pts,n_comp);
  for i=1:n_comp
      Z(:,i) = sum((X - repmat(C(i,:),n_pts,1)).^2, 2);
  end
  ll = ll + -.5* sum((U(:).^m).*Z(:));

  % cancel normalizing term in normal with fuzzy cluster prior
  % then add an additional symmetric dirichlet(alpha) distribution
  
  ll = ll + (alpha-1)*sum(log(U(:)));
  
  %sum of C lls in guassian with mu_c,siginv_c
  for i=1:n_comp
      z = C(i,:)-mu_c;
      ll = ll + -.5* z*siginv_c*z';
  end
  
end


function lls = u_lls(X,C,U,m)

  [n_pts,n_dim] = size(X);
  n_comp = size(C,1);

  lls = zeros(n_pts,1);
    
  % exponent terms
  Z = zeros(n_pts,n_comp);
  for i=1:n_comp
    Z(:,i) =  sum((X - repmat(C(i,:),n_pts,1)).^2, 2);
  end
  lls = lls + -.5 *sum((U.^m).*Z,2);

  % with fuzzy membership prior, base term of normal and 
  %  prior likelihoods cancel
    
end

function lls = u_lls_invx2(X,C,U,m)

  [n_pts,n_dim] = size(X);
  n_comp = size(C,1);

  lls = zeros(n_pts,1);
    
  % exponent terms
  Z = zeros(n_pts,n_comp);
  for i=1:n_comp
    Z(:,i) =  sum((X - repmat(C(i,:),n_pts,1)).^2, 2);
  end
  lls = lls + -.5 *sum((U.^m).*Z,2);

  % with fuzzy membership prior, base term of normal and
  %  first term of inv chi-sq prior cancel, leaves exponent term
  lls = lls - 1/2*sum(1./U,2);
    
end

function lls = u_lls_invgamma(X,C,U,m,beta)

  [n_pts,n_dim] = size(X);
  n_comp = size(C,1);

  lls = zeros(n_pts,1);
    
  % exponent terms
  Z = zeros(n_pts,n_comp);
  for i=1:n_comp
    Z(:,i) =  sum((X - repmat(C(i,:),n_pts,1)).^2, 2);
  end
  lls = lls + -.5 *sum((U.^m).*Z,2);

  % with fuzzy membership prior, base term of normal and
  %  first term of inv chi-sq prior cancel, leaves exponent term
  %beta = 0.00001;
  lls = lls - beta*sum(1./U,2);
    
end

function lls = u_lls_dirichlet(X,C,U,m,alpha)

  [n_pts,n_dim] = size(X);
  n_comp = size(C,1);

  lls = zeros(n_pts,1);
    
  % exponent terms
  Z = zeros(n_pts,n_comp);
  for i=1:n_comp
    Z(:,i) =  sum((X - repmat(C(i,:),n_pts,1)).^2, 2);
  end
  lls = lls + -.5 *sum((U.^m).*Z,2);

  % cancel normalizing term in normal with fuzzy cluster prior
  % then add an additional symmetric dirichlet(alpha) distribution
  
  lls = lls + (alpha-1)*sum(log(U),2);
    
end

function lls = u_lls_noprior(X,C,U,m)

  [n_pts,n_dim] = size(X);
  n_comp = size(C,1);

  lls = zeros(n_pts,1);
    
  % exponent terms
  Z = zeros(n_pts,n_comp);
  for i=1:n_comp
    Z(:,i) =  sum((X - repmat(C(i,:),n_pts,1)).^2, 2);
  end
  lls = lls + m*n_dim/2*sum(log(U),2) -.5 *sum((U.^m).*Z,2);

    
end


function drawfig(X,C,U,m,alpha,figno)

if ~exist('figno','var')
    figno = 12;
end

n_pts = size(X,1);
n_comp = size(C,1);

if n_pts > 10000
    return;
end

specs = {'r-','b-','g-','c-','m-','y-','k-'};
rgb = [U'; zeros(1,n_pts)];
        
figure(figno);
clf;
subplot(2,1,1);
scatter(X(:,1),X(:,2),[],U(:,1)'); %rgb');
hold on;
for i=1:n_comp
    text(C(i,1),C(i,2),num2str(i),'BackgroundColor','w','Color','k');
end
xlabel('D1');
ylabel('D2');
%title({'FCM+Dirichlet sampler',...
%    sprintf('m = %.1f, \\alpha = %.2f',m,alpha)});
title('membership in cluster 1');
axis equal;

subplot(2,1,2);
plot(U(:,1),specs{1});
hold on;
% for i=2:n_comp
%     plot(U(:,i),specs{i});
% end

xlabel('index');
ylabel('membership');
ylim([-0.01,1.01]);
drawnow;
        
end


