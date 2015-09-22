function [weights, theta] = perceptron_margin_train(x, y, n, k, rate)

    weights = zeros(n,1);
    theta = 0;
    gamma = 1;
    
    for runs = 1:20
        for i = 1:k
            label_pred = dot(weights,x(i,:)) + theta;
            if(y(i)*label_pred <= gamma)
                weights = weights + rate*y(i)*transpose(x(i,:));
                theta = theta + rate*y(i);
            end
        end
    end