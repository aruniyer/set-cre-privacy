function example1(Nseedmax, datasetname, outfile)
  % fid = fopen(outfile, 'w+');
	
  % set the list of theta values
  theta_list = [0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99];
  %theta_list = [0.1];

  sz_list = [100];
  % sz_list = [30 90 150 300 450 600];

  for theta_index = 1:length(theta_list)
    theta = theta_list(theta_index);
	
    for seed=1:Nseedmax
      for sz_index=1:length(sz_list)
        A = load(strcat('~/Workspace/Java/InfoLab/TDA/forpedrtest2/',datasetname,'_train_',num2str(theta),'_',num2str(seed),'_',num2str(sz_list(sz_index))));
        nf = size(A, 2);
        X = A(:, 1:nf - 1);
        Y = A(:,nf);
        id0 = Y == 0;
        id1 = Y == 1;
        Y(id0) = 1;
        Y(id1) = 2;
            
        A = load(strcat('~/Workspace/Java/InfoLab/TDA/forpedrtest2/',datasetname,'_test_',num2str(theta),'_',num2str(seed),'_',num2str(sz_list(sz_index))));
        xm = A(:, 1:nf - 1);
        ym = A(:, nf);
        id0 = ym == 0;
        id1 = ym == 1;
        ym(id0) = 1;
        ym(id1) = 2;
          
       	% estimate the class prior
        [p] = betaKMM_targetshift(X, Y, xm, ym, 1, 1, 1);
               
    	%Pest(theta_index, seed) = p;		
        distrib = [theta, 1 - theta];
        p = p.*distrib(Y)';
        est(1) = unique(p(Y == 1));
        est(2) = unique(p(Y == 2));
        est = est./sum(est);
        fprintf(fid, '%s,ZHANG,1,0,0,%d,0.5,0.5,%.2f,%.2f,%.2f,%.2f,0.0,0.0,N1,%d,%.2f,%f\n', datasetname, sz, theta(1), (1 - theta(1)), est(1), (1 - est(1)), seed, theta(1), (abs(theta(1) - est(1))));                
      end
    end	
  end
  fclose(fid);
end
