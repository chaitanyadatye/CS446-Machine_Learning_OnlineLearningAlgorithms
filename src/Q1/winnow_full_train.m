function mistakes = winnow_full_train(x,y,n,k,alpha, gamma)
    
    weights = ones(n,1);
    theta = -n;
    count_mistakes = 0;
    mistakes = [];
    
    for i = 1:k
        label_pred = dot(weights,x(i,:)) + theta;
        if(y(i)*label_pred <= gamma)
            if(y(i)*label_pred <= 0)
                count_mistakes = count_mistakes + 1;
            end
            for j = 1:n
                power = y(i)*x(i,j);
                weights(j) = weights(j)*(alpha^power);
            end
        end
        if(mod(i,100) == 0)
          mistakes = [mistakes, count_mistakes];
        end
    end