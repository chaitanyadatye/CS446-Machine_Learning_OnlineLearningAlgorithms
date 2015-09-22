function plot_graph(n, mistakes_p, mistakes_pm, mistakes_w, mistakes_wm, mistakes_a)
    
    x_axis_start = 0;
    x_axis_end = 50000;
    step_size = 100;
    
    x_axis = x_axis_start:step_size:x_axis_end;
    
    figure;
    plot(x_axis, mistakes_p, 'b', x_axis, mistakes_pm, 'r', ...
        x_axis, mistakes_w, 'k', x_axis, mistakes_wm, 'c', ...
        x_axis, mistakes_a, 'm' );
    
    if(n==1000)
        title('Plot of l=10, m=100, n=1000');
    else
        title('Plot of l=10, m=100, n=500');
    end
    legend({'Perceptron', 'Perceptron with Margin', 'Winnow', ...
        'Winnow with Margin', 'AdaGrad'}, 'Location', 'Northwest');
    xlabel('Number of examples');
    ylabel('Number of mistakes');
end
           