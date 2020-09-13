# importing the dataset
dataset = read.csv('Data.csv')
# dataset = dataset[,2:3]


# Splitting the dataset into training and test sets
library('caTools')
set.seed(123)
split = sample.split(dataset$Purchased , SplitRatio = 0.8)
training_set = subset(dataset , split == TRUE)
test_set = subset(dataset , split == FALSE)

# Feature scaling
# in R we should exclude the categorical features 
# if we include it this will cause an error
# training_set[ ,2:3] = scale(training_set[ ,2:3])
# test_set[ ,2:3] = scale(test_set[ ,2:3])
