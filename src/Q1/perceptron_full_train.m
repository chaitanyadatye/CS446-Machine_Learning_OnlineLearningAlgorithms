function mistakes = perceptron_full_train(x,y,n,k,rate, gamma)
    weights = zeros(n,1);
    theta = 0;
    count_mistakes = 0;
    mistakes = [];
    
    for i=1:k
        label_pred = dot(weights,x(i,:)) + theta;
        if(y(i)*label_pred <= gamma)
            if(y(i)*label_pred <= 0)
                count_mistakes = count_mistakes + 1;
            end
            weights = weights + rate*y(i)*transpose(x(i,:));
            theta = theta + rate*y(i);
        end
        if(mod(i,100) == 0)
          mistakes = [mistakes, count_mistakes];
        end   
    end
end