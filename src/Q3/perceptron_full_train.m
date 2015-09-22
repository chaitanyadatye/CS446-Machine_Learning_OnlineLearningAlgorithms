function [weights, theta] = perceptron_full_train(x,y,n,k,rate, gamma)
    weights = zeros(n,1);
    theta = 0;
    
    for runs = 1:20
        for i=1:k
            label_pred = dot(weights,x(i,:)) + theta;
            if(y(i)*label_pred <= gamma)
                weights = weights + rate*y(i)*transpose(x(i,:));
                theta = theta + rate*y(i);
            end
        end
    end
end