function [count_mistakes_pos, count_mistakes_neg] = gen_test(x, y, k, weights, theta)
    count_mistakes_pos = 0;
    count_mistakes_neg = 0;
    
    for i = 1:k
        val_pred = dot(weights,x(i,:)) + theta;
        if (val_pred <= 0)
            label_pred = -1;
        else
            label_pred = 1;
        end
        
        if(label_pred ~= y(i))
            if(y(i) == -1)
                count_mistakes_neg = count_mistakes_neg + 1;
            else
                count_mistakes_pos = count_mistakes_pos + 1;
            end
        end
    end
end
    
        
