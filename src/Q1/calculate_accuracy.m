function accuracy = calculate_accuracy(mistakes, k)
    
    accuracy = (1 - mistakes(end)/k)*100;
   