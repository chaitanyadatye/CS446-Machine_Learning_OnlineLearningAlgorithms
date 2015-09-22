function mistakes = gen_test(x, y, k, weights, theta)
    mistakes = [];
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
        
        if(mod(i,100) == 0)
            mistakes = [mistakes, count_mistakes];
        end
    end
    mistakes = [0,mistakes];
end
    
        
