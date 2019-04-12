# BayesianFuzzyClustering
Matlab implementation of the Bayesian Fuzzy Clustering algorithms.  Please cite this code when you use it: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2638099.svg)](https://doi.org/10.5281/zenodo.2638099)
Taylor Glenn, Alina Zare & Paul Gader. (2019, April 12). GatorSense/BayesianFuzzyClustering: Initial Release (Version v1.0). Zenodo. http://doi.org/10.5281/zenodo.2638099

***
See related paper, doi: 10.1109/TFUZZ.2014.2370676, http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=6955803

NOTE: If Bayesian Fuzzy Clustering is used in any publication or presentation, the following reference must be cited:  
<b>Glenn, T.; Zare, A.; Gader, P., "Bayesian Fuzzy Clustering," IEEE Transactions on Fuzzy Systems, vol.23, no.5, pp.1545-1561
doi: 10.1109/TFUZZ.2014.2370676</b>
***

Requirements:  
 This code uses the excellent Lightspeed and Fastfit toolboxes by Tom Minka:  
   Lightspeed toolbox - http://research.microsoft.com/en-us/um/people/minka/software/lightspeed/  
   Fastfit toolbox - http://research.microsoft.com/en-us/um/people/minka/software/fastfit/  

 This code also uses the Matlab Fuzzy Logic Toolbox for its fcm impleSeementation  

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

