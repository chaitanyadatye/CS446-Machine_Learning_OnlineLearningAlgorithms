function compare_algorithms(l,m,n,k_train,k_test)

    [train_y, train_x] = unba_gen(l,m,n,k_train,0.1);
    [test_y, test_x] = unba_gen(l,m,n,k_test,0.1);

    ten_perc = 0.1*k_train;
    
    d1_x_train = train_x(1:ten_perc, 1:n);
    d1_y_train = train_y(1:ten_perc);
    
    d2_x_train = train_x(ten_perc+1: ten_perc*2, 1:n);
    d2_y_train = train_y(ten_perc+1: ten_perc*2);    
    
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
    [mistakes_pm_pos, mistakes_pm_neg] = gen_test(test_x, test_y, k_test, weights_pm, theta_pm);
    accuracy_pm_pos_final = calculate_accuracy(mistakes_pm_pos, k_test/10);
    accuracy_pm_neg_final = calculate_accuracy(mistakes_pm_neg, 9*k_test/10);
    accuracy_pm_final = calculate_accuracy(mistakes_pm_neg+mistakes_pm_pos, k_test);
    display(accuracy_pm_pos_final);
    display(accuracy_pm_neg_final);
    display(accuracy_pm_final);
    display(best_pm_rate);
    
     %----------------------------------------------------------------------
    
    %This is the run for perceptron with margin modified
    rate_pmm = [1.5, 0.25, 0.03, 0.005, 0.001];
    best_pmm_accuracy = 0;
    best_pmm_rate = 0;
    for rate = rate_pmm
        [weights_pmm, theta_pmm] = perceptron_full_train_modified(d1_x_train, d1_y_train, n, ten_perc, rate, 1);
        mistakes_pmm = gen_test(d2_x_train, d2_y_train, ten_perc, weights_pmm, theta_pmm);
        accuracy_pmm = calculate_accuracy(mistakes_pmm, ten_perc);
        if(accuracy_pmm > best_pmm_accuracy)
            best_pmm_accuracy = accuracy_pmm;
            best_pmm_rate = rate;
        end
    end
    [weights_pmm, theta_pmm] = perceptron_full_train_modified(train_x, train_y, n, k_train, best_pmm_rate, 1);
    [mistakes_pmm_pos, mistakes_pmm_neg] = gen_test(test_x, test_y, k_test, weights_pmm, theta_pmm);
    accuracy_pmm_pos_final = calculate_accuracy(mistakes_pmm_pos, k_test/10);
    accuracy_pmm_neg_final = calculate_accuracy(mistakes_pmm_neg, 9*k_test/10);
    accuracy_pmm_final = calculate_accuracy(mistakes_pmm_neg+mistakes_pmm_pos, k_test);
    display(accuracy_pmm_pos_final);
    display(accuracy_pmm_neg_final);
    display(accuracy_pmm_final);
    display(best_pmm_rate);
