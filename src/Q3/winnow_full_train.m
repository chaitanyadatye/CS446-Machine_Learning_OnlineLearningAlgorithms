function weights = winnow_full_train(x,y,n,k,alpha, gamma)
    
    weights = ones(n,1);
    theta = -n;
    
    for runs = 1:20
        for i = 1:k
            label_pred = dot(weights,x(i,:)) + theta;
            if(y(i)*label_pred <= gamma)
                for j = 1:n
                    power = y(i)*x(i,j);
                    weights(j) = weights(j)*(alpha^power);
                end
            end
        end
    end