library(dplyr)
library(ggplot2)

getwd()
setwd("../../Psychology/Output/")

# In this script we follow the same steps as in the paper results

df1<-read.table('mart.csv',sep=',',header=TRUE, row.names = 1)
head(df1)
dim(df1)
summary(df1)

model <- lm(X0~X1,data=df1)
summary(model)

plot(df1$X0,df1$X1)
