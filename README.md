# BayesianFuzzyClustering
Matlab implementation of the Bayesian Fuzzy Clustering algorithms.  See related paper, doi: 10.1109/TFUZZ.2014.2370676

***
NOTE: If Bayesian Fuzzy Clustering is used in any publication or presentation, the following reference must be cited:  
<b>Glenn, T.; Zare, A.; Gader, P., "Bayesian Fuzzy Clustering," IEEE Transactions on Fuzzy Systems, vol.23, no.5, pp.1545-1561
doi: 10.1109/TFUZZ.2014.2370676</b>
***

Requirements:  
 This code uses the excellent Lightspeed and Fastfit toolboxes by Tom Minka:  
   Lightspeed toolbox - http://research.microsoft.com/en-us/um/people/minka/software/lightspeed/  
   Fastfit toolbox - http://research.microsoft.com/en-us/um/people/minka/software/fastfit/  

 This code also uses the Matlab Fuzzy Logic Toolbox for its fcm implementation  

Contents:  

bfc/  % code for Bayesian Fuzzy Clustering MCMC sampler  
|- bfc_params.m   % generate default parameters structure  
|- bfc_sampler.m  % bayesian fuzzy clustering sampler  
|- test.m         % test script - run this  

ibfc/ % code for Infinite Bayesian Fuzzy Clustering particle filter  
|- ibfc_pf_params.m % generate defaults parameters structure  
|- ibfc_pf.m        % IBFC particle filter  
|- u_update_ibfc.m  % membership update function  
|- c_update_ibfc.m  % cluster prototype update function  
|- ll_ibfc.m        % ibfc log-likelihood evaluation function  
|- test.m           % test script - run this  

