function count_mistakes = winnow_full_train(x,y,n,k,alpha, gamma)
    
    weights = ones(n,1);
    theta = -n;
    count_mistakes = 0;
    threshold = 0;
    
    for runs = 1:10
        if(threshold < 1000)
            for i = 1:k
                if(threshold < 1000)
                    label_pred = dot(weights,x(i,:)) + theta;
                    if(y(i)*label_pred <= 0)
                        count_mistakes = count_mistakes + 1;
                        threshold =0;
                    else
                        threshold = threshold + 1;
                    end
                    if(y(i)*label_pred <= gamma)
                        for j = 1:n
                            power = y(i)*x(i,j);
                            weights(j) = weights(j)*(alpha^power);
                        end
                    end
                else
                    break;
                end
            end
        else
            display('WinnowConverged');
            break;
        end
    end