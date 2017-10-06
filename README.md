
# Music Date Prediction
## MC886 - Machine Learning and Pattern Recognition 2S2017

### Giovani Nascimento Pereira - 168609
### Carlos Augusto Figueiredo Freire de Carvalho - 165484

## Introduction

This work was based on creating a Machine Learning model to predict the release year of a song, based on some features (timbre average and covariance) and using initially linear regression.
    
We worked using MatLAB (more specific, GNU Octave) to get the data and process it, without the use of any external libraries.

An important thing to remember before starting the Machine Learning models, is to remember to split your data into the test, validation and train groups. The test will never be used until the end, when we are going to compare the results of each selected model.


## Pure Linear Approach
 
The first approach was doing a simple linear regression using all of the data available (\cref{linearequation1}).
At the beginning, we had trouble considering the \textit{Learn Rate} (LR) because it was too big for the problem. This way our cost function would only raise higher and higher.
    
We finally got better result as LR = 10^{-8}, as an initial test, but we thought it was too small for the problem. This way, we decided to find a bigger LR so our model would converge faster. But we couldn't find it.
    
Anything bigger than 1*10^{-8} would make it diverge during the training.
But the error was bigger than 10^5.
 
 (...)