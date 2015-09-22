function [weights, theta] = perceptron_full_train_modified(x,y,n,k,rate, gamma)
    weights = zeros(n,1);
    theta = 0;
    num_of_pos_ex = histc(y,1);
    num_of_neg_ex = k - num_pos_ex;
    ratio = num_of_pos_ex/num_of_neg_ex;
    gamma_pos = gamma;
    gamma_neg = ratio*gamma;
    
    for runs = 1:20
        for i=1:k
            label_pred = dot(weights,x(i,:)) + theta;
            if(y(i) == 1)
                if(y(i)*label_pred <= gamma_pos)
                    weights = weights + rate*y(i)*transpose(x(i,:));
                    theta = theta + rate*y(i);
                end
            else
                if(y(i)*label_pred <= gamma_neg)
                    weights = weights + rate*y(i)*transpose(x(i,:));
                    theta = theta + rate*y(i);
                end
            end
        end
    end
end