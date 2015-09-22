function count_mistakes = perceptron_full_train(x,y,n,k,rate, gamma)
    weights = zeros(n,1);
    theta = 0;
    count_mistakes = 0;
    threshold = 0;

    for runs = 1:10
        if(threshold < 1000)
            for i=1:k
                if(threshold < 1000)
                    label_pred = dot(weights,x(i,:)) + theta;
                    if(y(i)*label_pred <= 0)
                        count_mistakes = count_mistakes + 1;
                        threshold = 0;
                    else
                        threshold = threshold + 1;
                    end
                    if(y(i)*label_pred <= gamma)
                        weights = weights + rate*y(i)*transpose(x(i,:));
                        theta = theta + rate*y(i);
                    end
                else
                    break;
                end
            end
        else
            display('PerceptronConverged');
            break;
        end
    end
end