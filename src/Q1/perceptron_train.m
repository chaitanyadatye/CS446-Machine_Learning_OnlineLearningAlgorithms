function [weights, theta] = perceptron_train(x,y,n,k)
    
    weights = zeros(n,1);
    theta = 0;
    
    for runs = 1:20
        for i = 1:k
            label_pred = dot(weights,x(i,:)) + theta;
            if(y(i)*label_pred <= 0)
                weights = weights + y(i)*transpose(x(i,:));
                theta = theta + y(i);
            end
        end
    end
        