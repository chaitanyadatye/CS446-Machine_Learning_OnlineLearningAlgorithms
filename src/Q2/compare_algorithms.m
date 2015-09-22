function compare_algorithms(l,m,k,noise)

    n_val = [40, 80, 120, 160, 200];
    perceptron_W = [0];
    perceptron_margin_W = [0];
    winnow_W = [0];
    winnow_margin_W = [0];
    adaGrad_W = [0];
    
    for n = n_val
        
        [y,x] = gen(l,m,n,k,noise);
        ten_perc = 0.1*k;

        d1_x = x(1:ten_perc, 1:n);
        d1_y = y(1:ten_perc);

        d2_x = x(ten_perc+1: ten_perc*2, 1:n);
        d2_y = y(ten_perc+1: ten_perc*2);    

        %This is the run for perceptron
        [weights_p, theta_p] = perceptron_train(d1_x, d1_y, n, ten_perc);
        mistakes = gen_test(d2_x, d2_y, ten_perc, weights_p, theta_p);
        %accuracy = calculate_accuracy(mistakes, ten_perc);
        mistakes_p = perceptron_full_train(x, y, n, k, 1, 0);
        %accuracy_perceptron = calculate_accuracy(mistakes_p, k);
        %display(accuracy_perceptron);
        display(mistakes_p);
        perceptron_W = [perceptron_W, mistakes_p];

        %----------------------------------------------------------------------

        %This is the run for perceptron with margin
        rate_pm = [1.5, 0.25, 0.03, 0.005, 0.001];
        best_pm_accuracy = 0;
        best_pm_rate = 0;
        best_pm_weights = [];
        best_pm_theta = 0;
        for rate = rate_pm

            [weights_pm, theta_pm] = perceptron_margin_train(d1_x, d1_y, n, ten_perc, rate);
            mistakes_pm = gen_test(d2_x, d2_y, ten_perc, weights_pm, theta_pm);
            accuracy_pm = calculate_accuracy(mistakes_pm, ten_perc);
            if(accuracy_pm > best_pm_accuracy)
                best_pm_accuracy = accuracy_pm;
                best_pm_rate = rate;
                best_pm_weights = weights_pm;
                best_pm_theta = theta_pm;
            end
        end
        mistakes_pm = perceptron_full_train(x, y, n, k, best_pm_rate, 1);
        %accuracy_pm_final = calculate_accuracy(mistakes_pm, k);
        %display(accuracy_pm_final);
        display(best_pm_rate);
        display(mistakes_pm);
        perceptron_margin_W = [perceptron_margin_W, mistakes_pm];
         
%     %     %----------------------------------------------------------------------
%     %     
%     %     %This is for Winnow
%     %     
        alpha_w = [1.1, 1.01, 1.005, 1.0005, 1.0001];
        theta_w = -n;
        best_w_accuracy = 0;
        best_w_weights = [];
        best_w_alpha = 0;

        for alpha = alpha_w
            weights_w = winnow_train(d1_x, d1_y, n, ten_perc, alpha, 0);
            mistakes_w = gen_test(d2_x, d2_y, ten_perc, weights_w, theta_w);
            accuracy_w = calculate_accuracy(mistakes_w, ten_perc);
            if(accuracy_w > best_w_accuracy)
                best_w_accuracy = accuracy_w;
                best_w_alpha = alpha;
                best_w_weights = weights_w;
            end
        end
        mistakes_w = winnow_full_train(x, y, n, k,best_w_alpha,0);
        %accuracy_w_final = calculate_accuracy(mistakes_w, k);
        %display(accuracy_w_final);
        display(best_w_alpha);
        display(mistakes_w);
        winnow_W = [winnow_W, mistakes_w];
        
%     %     %---------------------------------------------------------------------
%     %         
%     %     %This is for Winnow with margin
%     %     
        alpha_wm = [1.1, 1.01, 1.005, 1.0005, 1.0001];
        gamma_wm = [2.0, 0.3, 0.04, 0.006, 0.001];

        theta_wm = -n;
        best_wm_accuracy = 0;
        best_wm_weights = [];
        best_wm_alpha = 0;
        best_wm_gamma = 0;

        for alpha = alpha_wm
            for gamma = gamma_wm
                weights_wm = winnow_train(d1_x, d1_y, n, ten_perc, alpha, gamma);
                mistakes_wm = gen_test(d2_x, d2_y, ten_perc, weights_wm, theta_wm);
                accuracy_wm = calculate_accuracy(mistakes_wm, ten_perc);
                if(accuracy_wm > best_wm_accuracy)
                    best_wm_accuracy = accuracy_wm;
                    best_wm_alpha = alpha;
                    best_wm_gamma = gamma;
                    best_wm_weights = weights_wm;
                end
            end
        end
        mistakes_wm = winnow_full_train(x, y, n, k,best_wm_alpha,best_wm_gamma);
        %accuracy_wm_final = calculate_accuracy(mistakes_wm, k);
        %display(accuracy_wm_final);
        display(best_wm_alpha);
        display(best_wm_gamma);
        display(mistakes_wm);
        winnow_margin_W = [winnow_margin_W, mistakes_wm];
        
%     %     %---------------------------------------------------------------------
%     %     
%     %     %This is for AdaGrad
%     %     
        rate_a = [1.5, 0.25, 0.03, 0.005, 0.001];
        best_a_accuracy = 0;
        best_a_rate = 0;
        best_a_weights = [];
        best_a_theta = 0;
        for rate = rate_a
            [weights_a, theta_a] = adaGrad_train(d1_x, d1_y, n, ten_perc, rate);
            mistakes_a = gen_test(d2_x, d2_y, ten_perc, weights_a, theta_a);
            accuracy_a = calculate_accuracy(mistakes_a, ten_perc);
            if(accuracy_a > best_a_accuracy)
                best_a_accuracy = accuracy_a;
                best_a_rate = rate;
                best_a_weights = weights_a;
                best_a_theta = theta_a;
            end
        end
        mistakes_a = adaGrad_full_train(x, y, n, k, best_a_rate);
        %accuracy_a_final = calculate_accuracy(mistakes_a, k);
        %display(accuracy_a_final);
        display(best_a_rate);
        display(mistakes_a);
        adaGrad_W = [adaGrad_W, mistakes_a];
    end
%--------------------------------------------------------------------------

%Plot mistakes on graph

    n_val = [0, n_val];
    figure;
    plot(n_val, perceptron_W, '-b+', n_val,...
        perceptron_margin_W, '-ro', ...
        n_val, winnow_W, '-kd', n_val, winnow_margin_W, '-cs', ...
        n_val, adaGrad_W, '-m*');
    
    legend({'Perceptron', 'Perceptron with Margin', 'Winnow', ...
        'Winnow with Margin', 'AdaGrad'}, 'Location', 'Northwest');
    title('Mistakes of algorithms upon convergence plotted against number of features');
    xlabel('Number of features n');
    ylabel('Number of mistakes upon convergence');
    
    display(perceptron_W);
    display(perceptron_margin_W);
    display(winnow_W);
    display(winnow_margin_W);
    display(adaGrad_W);
    
    
    
    
    
    