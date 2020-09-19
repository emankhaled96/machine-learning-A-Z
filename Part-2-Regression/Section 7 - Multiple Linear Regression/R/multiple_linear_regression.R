# Multiple linear regression
# importing the dataset
dataset = read.csv('50_Startups.csv')


# encoding categorical data
dataset$State = factor(dataset$State,
                         levels = c('New York', 'California', 'Florida'),
                         labels = c(1,2,3))

# Splitting the dataset into training and test sets
library('caTools')
set.seed(123)
split = sample.split(dataset$Profit , SplitRatio = 0.8)
training_set = subset(dataset , split == TRUE)
test_set = subset(dataset , split == FALSE)

# Feature scaling
# we don't need to apply feature scaling manually because the library will take care of this
#fitting multiple linear regression to the training set
# we put in formula(~.) to make R know that we want the relation between profit and all the independent values ( x0,x1,x2...)
regressor = lm(formula = Profit ~.,
               data = training_set)
# predicting the test set
Y_pred = predict(regressor , newdata = test_set)

# Building the optimal model using backward elimination 

regressor = lm(Profit ~ R.D.Spend + Administration + Marketing.Spend+ State,
               data = dataset)
summary(regressor)

regressor = lm(Profit ~ R.D.Spend + Administration + Marketing.Spend,
               data = dataset)
summary(regressor)

regressor = lm(Profit ~ R.D.Spend  + Marketing.Spend,
               data = dataset)
summary(regressor)

regressor = lm(Profit ~ R.D.Spend  ,
               data = dataset)
summary(regressor)