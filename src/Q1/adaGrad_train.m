function [weights, theta] = adaGrad_train(x,y,n,k,rate)

    weights = zeros(n+1,1);
    grad_sum = zeros(1,n+1);
    theta_matrix = ones(k,1);
    x = horzcat(x, theta_matrix);
    for runs = 1:20
        for i = 1:k
            label_pred = dot(weights,x(i,:));
            grad_vec = zeros(1,n+1);
            if(y(i)*label_pred <= 1)
                grad_vec = -y(i)*x(i,:);
                grad_sum = grad_sum + (grad_vec.*grad_vec);
                for j = 1:n+1
                    if(grad_sum(j) ~= 0)
                        weights(j) = weights(j) + rate*y(i)*x(i,j)/sqrt(grad_sum(j));
                    end
                end
            end
        end
    end
    theta = weights(n+1);
    weights = weights(1:n);
end
                    
                
                
                
    