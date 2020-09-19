# importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Splitting the dataset into training and test sets
#library('caTools')
#set.seed(123)
#split = sample.split(dataset$Purchased , SplitRatio = 0.8)
#training_set = subset(dataset , split == TRUE)
#test_set = subset(dataset , split == FALSE)

# Feature scaling
# in R we should exclude the categorical features 
# if we include it this will cause an error
# training_set[ ,2:3] = scale(training_set[ ,2:3])
# test_set[ ,2:3] = scale(test_set[ ,2:3])


# fitting  regression model to the dataset

# enter your fitting code here

# predicting a new result with regression

y_pred = predict(regressor , data.frame(Level = 6.5))

# visualizing the regression model results

ggplot()+
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level , y = predict(regressor , newdata = dataset)),
            colour = 'blue')+
  ggtitle('Position Levels vs Salary (regression model)')+
  xlab('Position Levels')+
  ylab('Salary')
# visualizing the regression model results with higher resolution 
x_grid = seq(min(dataset$Level),max(dataset$Level),0.1)


ggplot()+
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level , y = predict(regressor , newdata = data.frame(Level = x_grid))),
            colour = 'blue')+
  ggtitle('Position Levels vs Salary (regression model)')+
  xlab('Position Levels')+
  ylab('Salary')




