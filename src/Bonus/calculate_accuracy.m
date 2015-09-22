function accuracy = calculate_accuracy(mistakes, k)
    
    accuracy = (1 - mistakes/k)*100;
   