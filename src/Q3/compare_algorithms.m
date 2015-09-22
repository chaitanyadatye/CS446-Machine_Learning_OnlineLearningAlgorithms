function compare_algorithms(l,m,n,k_train,k_test)

    [train_y, train_x] = gen(l,m,n,k_train,1);
    [test_y, test_x] = gen(l,m,n,k_test,0);

    ten_perc = 0.1*k_train;
    
    d1_x_train = train_x(1:ten_perc, 1:n);
    d1_y_train = train_y(1:ten_perc);
    
    d2_x_train = train_x(ten_perc+1: ten_perc*2, 1:n);
    d2_y_train = train_y(ten_perc+1: ten_perc*2);    
    
    %This is the run for perceptron
    %[weights_p, theta_p] = perceptron_train(d1_x_train, d1_y_train, n, ten_perc);
    %mistakes = gen_test(d2_x_train, d2_y_train, ten_perc, weights_p, theta_p);
    %accuracy = calculate_accuracy(mistakes, ten_perc);
    [weights_p, theta_p] = perceptron_full_train(train_x, train_y, n, k_train, 1, 0);
    mistakes_perceptron = gen_test(test_x, test_y, k_test, weights_p, theta_p); 
    accuracy_perceptron = calculate_accuracy(mistakes_perceptron, k_test);
    display(accuracy_perceptron);
    display(mistakes_perceptron);
  
    %----------------------------------------------------------------------
    
    %This is the run for perceptron with margin
    rate_pm = [1.5, 0.25, 0.03, 0.005, 0.001];
    best_pm_accuracy = 0;
    best_pm_rate = 0;
    for rate = rate_pm
        [weights_pm, theta_pm] = perceptron_full_train(d1_x_train, d1_y_train, n, ten_perc, rate, 1);
        mistakes_pm = gen_test(d2_x_train, d2_y_train, ten_perc, weights_pm, theta_pm);
        accuracy_pm = calculate_accuracy(mistakes_pm, ten_perc);
        if(accuracy_pm > best_pm_accuracy)
            best_pm_accuracy = accuracy_pm;
            best_pm_rate = rate;
        end
    end
    [weights_pm, theta_pm] = perceptron_full_train(train_x, train_y, n, k_train, best_pm_rate, 1);
    mistakes_pm = gen_test(test_x, test_y, k_test, weights_pm, theta_pm);
    accuracy_pm_final = calculate_accuracy(mistakes_pm, k_test);
    display(accuracy_pm_final);
    display(best_pm_rate);
    display(mistakes_pm);
%     
%     %----------------------------------------------------------------------
%     
%     %This is for Winnow
%     
    alpha_w = [1.1, 1.01, 1.005, 1.0005, 1.0001];
    theta_w = -n;
    best_w_accuracy = 0;
    best_w_alpha = 0;
    
    for alpha = alpha_w
        weights_w = winnow_full_train(d1_x_train, d1_y_train, n, ten_perc, alpha, 0);
        mistakes_w = gen_test(d2_x_train, d2_y_train, ten_perc, weights_w, theta_w);
        accuracy_w = calculate_accuracy(mistakes_w, ten_perc);
        if(accuracy_w > best_w_accuracy)
            best_w_accuracy = accuracy_w;
            best_w_alpha = alpha;
        end
    end
    weights_w = winnow_full_train(train_x, train_y, n, k_train, best_w_alpha, 0);
    mistakes_w = gen_test(test_x, test_y, k_test, weights_w, theta_w);
    accuracy_w_final = calculate_accuracy(mistakes_w, k_test);
    display(accuracy_w_final);
    display(best_w_alpha);
    display(mistakes_w);
% %     
% %     %---------------------------------------------------------------------
% %         
% %     %This is for Winnow with margin
% %     
    alpha_wm = [1.1, 1.01, 1.005, 1.0005, 1.0001];
    gamma_wm = [2.0, 0.3, 0.04, 0.006, 0.001];
    
    theta_wm = -n;
    best_wm_accuracy = 0;
    best_wm_alpha = 0;
    best_wm_gamma = 0;
    
    for alpha = alpha_wm
        for gamma = gamma_wm
            weights_wm = winnow_full_train(d1_x_train, d1_y_train, n, ten_perc, alpha, gamma);
            mistakes_wm = gen_test(d2_x_train, d2_y_train, ten_perc, weights_wm, theta_wm);
            accuracy_wm = calculate_accuracy(mistakes_wm, ten_perc);
            if(accuracy_wm > best_wm_accuracy)
                best_wm_accuracy = accuracy_wm;
                best_wm_alpha = alpha;
                best_wm_gamma = gamma;
            end
        end
    end
    weights_wm = winnow_full_train(train_x, train_y, n, k_train, best_wm_alpha, best_wm_gamma);
    mistakes_wm = gen_test(test_x, test_y, k_test, weights_wm, theta_wm);
    accuracy_wm_final = calculate_accuracy(mistakes_wm, k_test);
    display(accuracy_wm_final);
    display(best_wm_alpha);
    display(best_wm_gamma);
    display(mistakes_wm);
% %     
% %     %---------------------------------------------------------------------
% %     
% %     %This is for AdaGrad
    rate_a = [1.5, 0.25, 0.03, 0.005, 0.001];
    best_a_accuracy = 0;
    best_a_rate = 0;
    for rate = rate_a
        [weights_a, theta_a] = adaGrad_full_train(d1_x_train, d1_y_train, n, ten_perc, rate);
        mistakes_a = gen_test(d2_x_train, d2_y_train, ten_perc, weights_a, theta_a);
        accuracy_a = calculate_accuracy(mistakes_a, ten_perc);
        if(accuracy_a > best_a_accuracy)
            best_a_accuracy = accuracy_a;
            best_a_rate = rate;
        end
    end
    [weights_a, theta_a] = adaGrad_full_train(train_x, train_y, n, k_train, best_a_rate);
    mistakes_a = gen_test(test_x, test_y, k_test, weights_a, theta_a);
    accuracy_a_final = calculate_accuracy(mistakes_a, k_test);
    display(accuracy_a_final);
    display(best_a_rate);
    display(mistakes_a);

    
    
    