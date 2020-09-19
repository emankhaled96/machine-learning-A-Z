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
# fitting linear regression to the dataset
lin_reg = lm(formula =Salary ~.,
             data = dataset)

# fitting polynomial regression to the dataset
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4

poly_reg = lm(formula =Salary~. ,
              data =dataset )

# visualizing the linear regression results
install.packages('ggplot2')
library(ggplot2)
ggplot()+
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level , y = predict(lin_reg , newdata = dataset)),
            colour = 'blue')+
  ggtitle('Position Levels vs Salary (linear regression)')+
  xlab('Position Levels')+
  ylab('Salary')

# visualizing the polynomial regression results

ggplot()+
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level , y = predict(poly_reg , newdata = dataset)),
            colour = 'blue')+
  ggtitle('Position Levels vs Salary (polynomial regression)')+
  xlab('Position Levels')+
  ylab('Salary')


# predicting a new result with linear regression

y_pred = predict(lin_reg , data.frame(Level = 6.5))


# predicting a new result with polynomial regression

y_pred = predict(poly_reg , data.frame(Level = 6.5,
                                       Level2 = 6.5^2,
                                       Level3 = 6.5^3,
                                       Level4 = 6.5^4))