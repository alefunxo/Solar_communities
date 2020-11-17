install.packages("GiniWegNeg")
library(GiniWegNeg)
help(Gini_RSV)
# test_bill contains the bills of every single household and its type (PV, PV_batt, No)
d<-read.table('test_bill.csv',sep=',',header=TRUE,stringsAsFactors = FALSE)
# if every hh has the same weight:
Gini_RSV(d$SC*-1)
Gini_RSV(d$P2P*-1)

# if we give hh with PV and battery higher weight than PV and PV higher weight than those w/o devices:
d$weight <- ifelse(d$type == 'PV_batt', 4, ifelse(d$type == 'PV', 2, 1))

Gini_RSV(d$SC*-1,d$weight)
Gini_RSV(d$P2P*-1,d$weight)

# To have lower Ginis in P2P the weights should be very high for PV_batt
d$weight <- ifelse(d$type == 'PV_batt', 12, ifelse(d$type == 'PV', 2, 1))
Gini_RSV(d$SC*-1,d$weight)
Gini_RSV(d$P2P*-1,d$weight)


y<-d$SC*-1
w<-d$weight

Gini_RSV <-
  function(y,w=rep(1,length(y)))
  {



dataset<-cbind(y,w)
ord_y<-order(y)
  
dataset_ord<-dataset[ord_y,] 
y<-dataset_ord[,1]
w<-dataset_ord[,2]
N<-sum(w)
yw<-y*w
C_i<-cumsum(w)
num_1<-sum(yw*C_i)
num_2<-sum(yw)
num_3<-sum(yw*w)

G_num<-(2/N^2)*num_1-(1/N)*num_2-(1/N^2)*num_3
t_neg<-subset(yw,yw<=0)
  T_neg<-sum(t_neg)
  T_pos<-sum(yw)+abs(T_neg)
  n_RSV<-(2*(T_pos+(abs(T_neg)))/N)
  mean_RSV<-(n_RSV/2)
  G_RSV<-(1/mean_RSV)*G_num
  list(GINI_RSV=G_RSV)
