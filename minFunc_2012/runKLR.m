function runKLR(Nseedmax, datasetname, outfile)

    fid = fopen(outfile, 'w+');

	% set the list of theta values
	theta_list = [0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99];

    for theta_index = 1:length(theta_list)
		theta = theta_list(theta_index);
	
		for seed=1:Nseedmax
            A = load(strcat('forpedrtest/',datasetname,'_train_',num2str(theta),'_',num2str(seed)));
            nf = size(A, 2);
            X = A(:, 1:nf - 1);
            y = A(:,nf);
            y(y == 0) = -1;
            
            A = load(strcat('forpedrtest/',datasetname,'_test_',num2str(theta),'_',num2str(seed)));
            Xt = A(:, 1:nf - 1);
            yt = A(:,nf); 
            yt(yt == 0) = -1;
            
			[uRBF, preds, probs] = KLR(X, y, Xt,yt, 1, 1000);
            
            estthetan = 1 - length(preds(find(preds == 1)))/length(preds);
            
            fprintf(fid, '%s,KLR,1,0,0,0.5,0.5,%.2f,%.2f,%.2f,%.2f,0.0,0.0,N1,%d,%.2f,%f\n', datasetname, theta(1), (1 - theta(1)), estthetan, (1 - estthetan), seed, theta(1), (abs(theta(1) - estthetan)));
		end	
    end
	
    fclose(fid);
end