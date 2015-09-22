function count_mistakes = gen_test(x, y, k, weights, theta)
    count_mistakes = 0;
    
    for i = 1:k
        val_pred = dot(weights,x(i,:)) + theta;
        if (val_pred <= 0)
            label_pred = -1;
        else
            label_pred = 1;
        end
        
        if(label_pred ~= y(i))
            count_mistakes = count_mistakes + 1;
        end
    end
end
    
        
