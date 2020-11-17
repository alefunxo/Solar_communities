###################Load#####################################
library(dplyr)
library(ggplot2)
library(ggpubr)
library(ggtern)
library(ggrepel)
library(adklakedata)
library(ade4)
library(adegraphics)
library(grid)
library(GiniWegNeg)
library(forcats)


library(gtools)
addUnits <- function(n) {
  labels <- ifelse(n < 1000, n,  # less than thousands
                   ifelse(n < 1e6, paste0(round(n/1e3), ''),  # in thousands
                          ifelse(n < 1e9, paste0(round(n/1e6), 'M'),  # in millions
                                 ifelse(n < 1e12, paste0(round(n/1e9), 'B'), # in billions
                                        ifelse(n < 1e15, paste0(round(n/1e12), 'T'), # in trillions
                                               'too big!'
                                        )))))
  return(labels)
}
##########init#####################
#read the dataframe
setwd("/data/home/alejandropena/Psychology/src")
d<-read.table('df_core.csv',sep=',',header=TRUE,stringsAsFactors = FALSE)

head(d)
dim(d)


df<-read.table('df_50_50_DE_0.25_100_False_1000.csv',sep=',',header=TRUE,stringsAsFactors = FALSE)
head(df)
dim(df)
#centroids <- aggregate(cbind(df_c3$SS,df_c3$TSC)~df_c3$conf+df_c3$ht,df_c3,median)
df_1 <- subset(df,df$index!='No')


#################Survey#####################
p_sur <- ggplot(d, aes(x=Price/100,y=Sell*100, group=sp,fill=factor(sp)))
p_sur <- p_sur + facet_grid(~SOC,labeller = as_labeller(c("30"= "SOC\n30%","60"= "SOC\n60%",
                                              "90"= "SOC\n90%")))
p_sur <- p_sur + geom_bar(stat="identity",position="dodge")
p_sur <- p_sur + theme_bw()
p_sur <- p_sur + scale_x_continuous(breaks=seq(0.04,0.28,0.08),limits=c(0,0.32))
p_sur <- p_sur + scale_y_continuous(limits=c(0,100))
p_sur <- p_sur + labs(fill="Surplus time:")
p_sur <- p_sur + scale_fill_manual(values = c("#F8766D", "#00BFC4"),
                                     labels=c('Within 12h','In more\nthan 12h'))
p_sur <- p_sur + ylab('Decision to sell\n[%]')
p_sur <- p_sur + xlab('Price[€/kWh]')

p_sur <- p_sur + theme(legend.text=element_text(size=14),
                       legend.title = element_text(size=16,face="bold"))
p_sur <- p_sur + theme(axis.text=element_text(size=14),
                       axis.title=element_text(size=14,face="bold"))
p_sur
ggsave('../Img/Selling.pdf',plot=p_sur,width=15, height=8.5,
       encoding = "ISOLatin9.enc")

############# FIGURE 1A#########################
p_tss <- ggplot(df_1, aes(x=SCR_hh, y=SSR_hh,shape=factor(index), fill=factor(index),colour=factor(Comm)))
p_tss <- p_tss + geom_point(size=3)
p_tss <- p_tss + theme_bw()
p_tss <- p_tss + scale_shape_manual(values =c(16, 17),
                                    labels=c("Prosumer with\nPV","Prosumer with\nPV & Battery"))
p_tss <- p_tss + scale_colour_manual(values = alpha(c("#F8766D", "#00BFC4"),.3),
                                     labels=c('P2P','Self-consumption'))
p_tss <- p_tss + labs(shape="Type of prosumer:",fill="Type of prosumer:" ,colour="Community:")
p_tss <- p_tss + scale_y_continuous(limits=c(0,100),expand = c(0,0))
p_tss <- p_tss + scale_x_continuous(limits=c(0,100),expand = c(0,0))
p_tss <- p_tss + theme(plot.margin = unit(c(0.5,1,0,0), "cm"))
#p_tss <- p_tss + coord_cartesian(xlim=c(0,100))
p_tss <- p_tss + xlab('SC [%]')
p_tss <- p_tss + ylab('SS\n[%]')
p_tss <- p_tss + geom_abline(aes(intercept = 0,slope=1))
p_tss <- p_tss + scale_fill_manual(values = c("#00AFBB", "#E7B800", "#FC4E07")) 

p_tss <- p_tss + theme(legend.position="bottom",legend.text=element_text(size=14),
                       legend.title = element_text(size=16,face="bold"))
p_tss <- p_tss + labs(subtitle = "A) Household level",face="bold")
p_tss <- p_tss + theme(plot.subtitle = element_text(hjust = 0.5,size=16,face="bold"))
p_tss <- p_tss + theme(axis.text=element_text(size=14),
                       axis.title=element_text(size=14,face="bold"))
p_tss <- p_tss + guides(colour=FALSE, fill=FALSE)
p_tss <- p_tss + geom_text(aes(x=10, label="Net producer", y=90), colour="black",alpha=0.8)
p_tss <- p_tss + geom_text(aes(x=90, label="Net consumer", y=10), colour="black",alpha=0.8)
#p_tss                    
#p_tss <- p_tss + facet_grid(~Comm)

############# FIGURE 1B#########################
p_tss2 <- ggplot(df, aes(x=df$SCR, y=df$SSR,colour=factor(Comm)))#,colour=factor(Comm)))
p_tss2 <- p_tss2 + theme_bw()
p_tss2 <- p_tss2 + geom_point(size=3,shape=15)#aes(color = factor(index)))#,shape=factor(df$index)))
p_tss2 <- p_tss2 + labs(colour="Community:")
p_tss2 <- p_tss2 + scale_colour_manual(values = c("#F8766D", "#00BFC4"),
                                     labels=c('P2P','Self-\nconsumption'))
p_tss2 <- p_tss2 + scale_y_continuous(limits=c(0,100),expand = c(0,0))
p_tss2 <- p_tss2 + scale_x_continuous(limits=c(0,100),expand = c(0,0))
p_tss2 <- p_tss2 + theme(plot.margin = unit(c(0.5,1,0,0), "cm"))
p_tss2 <- p_tss2 + coord_cartesian(xlim=c(0,100))
p_tss2 <- p_tss2 + xlab('SC [%]')
p_tss2 <- p_tss2 + ylab('SS\n[%]')
p_tss2 <- p_tss2 + geom_abline(aes(intercept = 0,slope=1))
p_tss2 <- p_tss2 + scale_fill_manual(values = c("#00AFBB", "#E7
                                                B800", "#FC4E07")) 
p_tss2 <- p_tss2 + theme(legend.position="bottom",legend.text=element_text(size=14),
                         legend.title = element_text(size=16,face="bold"))
p_tss2 <- p_tss2 + labs(subtitle = "B) Community level",face="bold")
p_tss2 <- p_tss2 + geom_text(aes(x=10, label="Net producer", y=90), colour="black",alpha=0.8)
p_tss2 <- p_tss2 + geom_text(aes(x=90, label="Net consumer", y=10), colour="black",alpha=0.8)

p_tss2 <- p_tss2 + theme(plot.subtitle = element_text(hjust = 0.5,size=16,face="bold"))
p_tss2 <- p_tss2 + theme(axis.text=element_text(size=14),
                       axis.title=element_text(size=14,face="bold"))
grid.arrange(p_tss,p_tss2, ncol=2)

a<-tapply(subset(df,df$Comm=='P2P')$bill_hh, subset(df,df$Comm=='P2P')$index, median)
b<-tapply(subset(df,df$Comm=='SC')$bill_hh, subset(df,df$Comm=='SC')$index, median)
a-b
(a-b)/b
a<-tapply(subset(df,df$Comm=='P2P')$bill, subset(df,df$Comm=='P2P')$index, median)
b<-tapply(subset(df,df$Comm=='SC')$bill, subset(df,df$Comm=='SC')$index, median)
b
a-b
############# FIGURE 1C#########################
p_bill <- ggboxplot(df, x= "Comm",y="bill_hh",fill="Comm")+facet_grid(~index,
                      labeller=as_labeller(c("No"="Consumer","PV"="Prosumer with PV",
                                             "PV_batt"="Prosumer with PV and battery")))

p_bill <- p_bill + stat_compare_means(label.y=800)

p_bill <- p_bill + scale_x_discrete(labels=c('P2P','Self-\nconsumption'))

p_bill <- p_bill + scale_colour_manual(values = alpha(c("#F8766D", "#00BFC4"),.3),
                                     labels=c('P2P','Self-consumption'))
p_bill <- p_bill + labs(shape="Type of prosumer:",fill="Type of prosumer:" ,colour="Community:")

p_bill <- p_bill + ylab('Bill\n[€ p.a.]')
p_bill <- p_bill + xlab('Type of member')
p_bill <- p_bill + theme_bw()
p_bill <- p_bill + theme(plot.margin = unit(c(0.5,1,0,0), "cm"))

p_bill <- p_bill + theme(legend.position="bottom",legend.text=element_text(size=14),
                       legend.title = element_text(size=16,face="bold"))
p_bill <- p_bill + labs(subtitle = "C) Household level",face="bold")
p_bill <- p_bill + theme(plot.subtitle = element_text(hjust = 0.5,size=16,face="bold"))
p_bill <- p_bill + theme(axis.text=element_text(size=14),
                       axis.title=element_text(size=14,face="bold"))
p_bill <- p_bill + guides(colour=FALSE, fill=FALSE)
#p_bill

############# FIGURE 1D#########################
p_bill2 <- ggboxplot(df, x= "Comm",y= "bill" ,fill="Comm")
p_bill2 <- p_bill2 + stat_compare_means(label.y=40000,label.x=1.35)
#p_bill2 <- p_bill2 + geom_boxplot()
p_bill2 <- p_bill2 + theme_bw()
p_bill2 <- p_bill2 + scale_x_discrete(labels=c("P2P","Self-consumption"))
p_bill2 <- p_bill2 + labs(shape="Type of prosumer:",fill="Type of prosumer:" ,colour="Community:")
p_bill2 <- p_bill2 + ylab('Bill\n[k€ p.a.]')
p_bill2 <- p_bill2 + xlab('Type of community')
p_bill2 <- p_bill2 + scale_y_continuous(labels = addUnits,limits = c(35000,95000))
p_bill2 <- p_bill2 + theme(plot.margin = unit(c(0.5,1,0,0), "cm"))

p_bill2 <- p_bill2 + theme(legend.position="bottom",legend.text=element_text(size=14),
                         legend.title = element_text(size=16,face="bold"))
p_bill2 <- p_bill2 + labs(subtitle = "D) Community level",face="bold")
p_bill2 <- p_bill2 + theme(plot.subtitle = element_text(hjust = 0.5,size=16,face="bold"))
p_bill2 <- p_bill2 + theme(axis.text=element_text(size=14),
                         axis.title=element_text(size=14,face="bold"))
p_bill2 <- p_bill2 + guides(colour=FALSE, fill=FALSE)
#p_bill2
grid.arrange(p_tss,p_tss2,p_bill,p_bill2,nrow=2, ncol=2)


ggsave('../Img/P3.pdf',plot=grid.arrange(p_tss,p_tss2,p_bill,p_bill2,nrow=2, ncol=2),width=15, height=8.5,
       encoding = "ISOLatin9.enc")


a<-tapply(subset(df,df$Comm=='P2P')$Demand_peak, subset(df,df$Comm=='P2P')$index, median)
b<-tapply(subset(df,df$Comm=='SC')$Demand_peak, subset(df,df$Comm=='SC')$index, median)
(a-b)/b
a<-tapply(subset(df,df$Comm=='P2P')$Inj_peak, subset(df,df$Comm=='P2P')$index, median)
b<-tapply(subset(df,df$Comm=='SC')$Inj_peak, subset(df,df$Comm=='SC')$index, median)
(a-b)/b


############# FIGURE 2A#########################

df2<-read.table('peak_50_50_DE_0.25_100_1000_tidy.csv',sep=',',header=TRUE,stringsAsFactors = FALSE)
df3<-read.table('week_50_50_DE_0.25_100.csv',sep=',',header=TRUE,stringsAsFactors = FALSE)
head(df2)

p_peak <- ggboxplot(df2, x = "Comm", y = "Power",fill= "Comm")+ facet_grid(~season)
#p_peak <- p_peak + geom_boxplot(aes(x=factor(Comm),y=Power,fill=factor(Comm)))+facet_grid(~season)
p_peak <- p_peak + stat_compare_means(label.y=0,size=3)
p_peak <- p_peak + theme_bw()
p_peak <- p_peak + scale_x_discrete(labels=c("P2P","Self-\nconsumption"))
p_peak <- p_peak + labs(fill="Community:")
p_peak <- p_peak + ylab('Peak-to-peak\n difference [kW]')
p_peak <- p_peak + xlab('Type of community')
p_peak <- p_peak + ylim(0,NA)
p_peak <- p_peak + labs(subtitle = "C) Peak-to-peak seasonal effect",face="bold")
p_peak <- p_peak + theme(plot.subtitle = element_text(hjust = 0.5,size=16,face="bold"))

p_peak <- p_peak + theme(legend.position="left",legend.text=element_text(size=14),
                           legend.title = element_text(size=16,face="bold"))
p_peak <- p_peak + theme(axis.text=element_text(size=14),
                           axis.title=element_text(size=14,face="bold"))
p_peak <- p_peak + theme(plot.margin = unit(c(0.5,0.5,0,0), "cm"))

############# FIGURE 2B#########################
df4 <- subset(df3,(df3$seed==167))
df5<-rbind(df4[c(144:168),],df4[c(1:144),])
rownames(df5) <- NULL
df5$Y<-as.numeric(rownames(df5))-1
library(scales)
cols = hue_pal()(12)
#show_col(hue_pal()(12))

al=1
p_peak2 <- ggplot(df5)
p_peak2 <- p_peak2 + theme_bw()
p_peak2 <- p_peak2 + geom_line(aes(x=Y,y=sc,colour="#F8766D"))
p_peak2 <- p_peak2 + geom_line(aes(x=Y,y=p2p,colour="#00BFC4"))
p_peak2 <- p_peak2 + scale_colour_manual(values = c("#F8766D", "#00BFC4"),
                                         labels=c("P2P","Self-consumption" ))
#Expand is the magic word!
p_peak2 <- p_peak2 + scale_x_continuous(breaks=seq(0,168,24),limits=c(0,172),expand = c(0, 0))
p_peak2 <- p_peak2 + scale_y_continuous(limits=c(-125,70),expand = c(0, 0))


p_peak2 <- p_peak2 + labs(colour="Community:")
p_peak2 <- p_peak2 + ylab('Power [kW]')
p_peak2 <- p_peak2 + xlab('Time [hours]')
p_peak2 <- p_peak2 + coord_cartesian(xlim=c(0,172))
p_peak2 <- p_peak2 + labs(subtitle = "D) Average grid exchange in one week",face="bold")
p_peak2 <- p_peak2 + theme(plot.subtitle = element_text(hjust = 0.5,size=16,face="bold"))
p_peak2 <- p_peak2 + theme(legend.position="bottom",legend.text=element_text(size=14),
                         legend.title = element_text(size=16,face="bold"))
p_peak2 <- p_peak2 + theme(axis.text=element_text(size=14),
                         axis.title=element_text(size=14,face="bold"))
p_peak2 <- p_peak2 + theme(plot.margin = unit(c(0.5,0.5,0,0), "cm"))

p_peak2 <- p_peak2 + annotate('segment',x=62, xend=62,
                 y = -120, yend = 57.5,colour="blue",alpha=0.5,size=1)
p_peak2 <- p_peak2 + annotate('rect',xmin=168, xmax=172,
                              ymin = 0, ymax = Inf,fill=cols[5],alpha=0.5)
p_peak2 <- p_peak2 + annotate('rect',xmin=168, xmax=172,
                              ymin = -Inf, ymax = 0,fill=cols[10])#,alpha=0.5)
p_peak2 <- p_peak2 + geom_hline(yintercept = 0)
p_peak2 <- p_peak2 + geom_text(aes(x=169.5, label="Import", y=35), colour="black",alpha=0.8,
                               angle=90)
p_peak2 <- p_peak2 + geom_text(aes(x=169.5, label="Export", y=-65), colour="black",alpha=0.8,
                               angle=90)
p_peak2 <- p_peak2 + geom_text(aes(x=63.5, label="peak-to-peak difference", y=0), colour="gray",alpha=0.8,
                               angle=90)
#p_peak2

############# FIGURE 2C#########################

p_peak3 <- ggboxplot(df, x= "Comm",y="Demand_peak",fill="Comm")

p_peak3 <- p_peak3 + stat_compare_means(label.x=1.3,label.y=100)

p_peak3 <- p_peak3 + scale_x_discrete(labels=c('P2P','Self-\nconsumption'))

p_peak3 <- p_peak3 + scale_colour_manual(values = alpha(c("#F8766D", "#00BFC4"),.3),
                                       labels=c('P2P','Self-consumption'))
p_peak3 <- p_peak3 + labs(shape="Type of prosumer:",fill="Type of prosumer:" ,colour="Community:")

p_peak3 <- p_peak3 + ylab('Power\n[kW]')
p_peak3 <- p_peak3 + xlab('Type of community')
p_peak3 <- p_peak3 + theme_bw()
p_peak3 <- p_peak3 + scale_y_continuous(limits=c(100,300))
p_peak3 <- p_peak3 + theme(plot.margin = unit(c(0.5,1,0,0), "cm"))

p_peak3 <- p_peak3 + theme(legend.position="bottom",legend.text=element_text(size=14),
                         legend.title = element_text(size=16,face="bold"))
p_peak3 <- p_peak3 + labs(subtitle = "A) Maximum import peak",face="bold")
p_peak3 <- p_peak3 + theme(plot.subtitle = element_text(hjust = 0.5,size=16,face="bold"))
p_peak3 <- p_peak3 + theme(axis.text=element_text(size=14),
                         axis.title=element_text(size=14,face="bold"))
p_peak3 <- p_peak3 + guides(colour=FALSE, fill=FALSE)
p_peak3
############# FIGURE 2D#########################
p_peak4 <- ggboxplot(df, x= "Comm",y="Inj_peak",fill="Comm")
p_peak4 <- p_peak4 + stat_compare_means(label.x=1.3,label.y=100)

p_peak4 <- p_peak4 + scale_x_discrete(labels=c('P2P','Self-\nconsumption'))

p_peak4 <- p_peak4 + scale_colour_manual(values = alpha(c("#F8766D", "#00BFC4"),.3),
                                       labels=c('P2P','Self-consumption'))
p_peak4 <- p_peak4 + labs(shape="Type of prosumer:",fill="Type of prosumer:" ,colour="Community:")

p_peak4 <- p_peak4 + ylab('Power\n[kW]')
p_peak4 <- p_peak4 + xlab('Type of community')
p_peak4 <- p_peak4 + theme_bw()
p_peak4 <- p_peak4 + scale_y_continuous(limits=c(100,300))
p_peak4 <- p_peak4 + theme(plot.margin = unit(c(0.5,1,0,0), "cm"))

p_peak4 <- p_peak4 + theme(legend.position="bottom",legend.text=element_text(size=14),
                         legend.title = element_text(size=16,face="bold"))
p_peak4 <- p_peak4 + labs(subtitle = "B) Maximum export peak",face="bold")
p_peak4 <- p_peak4 + theme(plot.subtitle = element_text(hjust = 0.5,size=16,face="bold"))
p_peak4 <- p_peak4 + theme(axis.text=element_text(size=14),
                         axis.title=element_text(size=14,face="bold"))
p_peak4 <- p_peak4 + guides(colour=FALSE, fill=FALSE)
p_peak4
lay <- rbind(c(3,4,2,2),
             c(1,1,1,1))
grid.arrange(p_peak2,p_peak,p_peak3,p_peak4, layout_matrix = lay)
ggsave('../Img/Power.pdf',plot=grid.arrange(p_peak2,p_peak,p_peak3,p_peak4, layout_matrix = lay),
       width=15, height=8.5,
       encoding = "ISOLatin9.enc")

#######################tests######
median(subset(df,df$Comm=='P2P')$Demand_peak)-median(subset(df,df$Comm=='SC')$Demand_peak)
median(subset(df,df$Comm=='P2P')$Inj_peak)-median(subset(df,df$Comm=='SC')$Inj_peak)

colnames(df)
df$total_gen#MWh
df$Total_load#MW
median(df$total_gen)#MWh
df$Total_load/4#MWh
1500*50*6/1000
1-df$EPARI
df$ADMD
median(subset(df,df$Comm=='P2P')$ADME)-median(subset(df,df$Comm=='SC')$ADME)#max_exp/number of hh
median(subset(df,df$Comm=='P2P')$ADMD)-median(subset(df,df$Comm=='SC')$ADMD)#max_imp/number of hh
median(1-(subset(df,df$Comm=='P2P')$EPARI)/median(subset(df,df$Comm=='SC')$EPARI))#max_exp/number of hh

df2_fall<-subset(df2,df2$season=='Fall')
df2_spring<-subset(df2,df2$season=='Spring')
df2_winter<-subset(df2,df2$season=='Winter')
df2_summer<-subset(df2,df2$season=='Summer')

shapiro.test(sample(df2_fall$Power,5000))
shapiro.test(sample(df2_spring$Power,5000))
shapiro.test(sample(df2_winter$Power,5000))
shapiro.test(sample(df2_summer$Power,5000))

pairwise.wilcox.test(df2_fall$Power,df2_fall$Comm,paired=FALSE)
pairwise.wilcox.test(df2_spring$Power,df2_spring$Comm,paired=FALSE)
pairwise.wilcox.test(df2_winter$Power,df2_winter$Comm,paired=FALSE)
pairwise.wilcox.test(df2_summer$Power,df2_summer$Comm,paired=FALSE)

###############################FIGURE 3A######################################
d<-read.table('Optim_trading.csv',sep=',',header=TRUE,stringsAsFactors = FALSE)

scr_hh_sc <- median(subset(d,(d$Comm=='SC')&(d$index=='PV_batt'))$SCR_hh)
ssr_hh_sc <- median(subset(d,(d$Comm=='SC')&(d$index=='PV_batt'))$SSR_hh)
bill_hh_sc <- median(subset(d,(d$Comm=='SC')&(d$index=='PV_batt'))$bill_hh)

d<-subset(d,(d$Comm=='P2P')&(d$index=='PV_batt'))
d$trading<-ordered(d$trading, levels = c("High", "Normal", "Low"))
d.m <- melt(select(d,-X), id.var = c("trading","index", "Comm"))
head(d.m)
unique(d.m$variable)

dummy1 <- data.frame("variable" = c("SCR_hh","SSR_hh","bill_hh"), Z = c(scr_hh_sc,ssr_hh_sc, bill_hh_sc))

p_hh <- ggboxplot(d.m, x= "trading",y="value",fill="trading")
p_hh <- p_hh + geom_hline(data=dummy1, aes(yintercept = Z),colour="red",linetype="longdash")
p_hh <- p_hh + facet_wrap(~variable, scales="free_y",
                          labeller=as_labeller(c("SCR_hh"="SC [%]","SSR_hh"="SS [%]","bill_hh"="Bill [€]")))
p_hh <- p_hh + stat_compare_means(label.x=1.5)

p_hh <- p_hh + scale_fill_discrete(name =  'Dark2')
p_hh <- p_hh + labs(shape="Type of prosumer:",fill="Type of prosumer:" ,colour="Community:")

p_hh <- p_hh + theme_bw()
p_hh <- p_hh + theme(plot.margin = unit(c(0.5,1,0,0), "cm"))
p_hh <- p_hh + xlab('Amount of trading in P2P community')
p_hh <- p_hh + theme(legend.position="bottom",legend.text=element_text(size=14),
                     legend.title = element_text(size=16,face="bold"))
p_hh <- p_hh + labs(subtitle = "A) Households with PV and battery",face="bold")
p_hh <- p_hh + theme(plot.subtitle = element_text(hjust = 0.5,size=16,face="bold"))
p_hh <- p_hh + theme(axis.text=element_text(size=14),
                     axis.title=element_text(size=14,face="bold"))
p_hh <- p_hh + guides(colour=FALSE, fill=FALSE)
#p_hh


###########################FIGURE 3B###################################
d2<-read.table('Optim_trading_comm.csv',sep=',',header=TRUE,stringsAsFactors = TRUE)

scr_sc <- median(subset(d2,(d2$Comm=='SC'))$SCR)
ssr_sc <- median(subset(d2,(d2$Comm=='SC'))$SSR)
bill_sc <- median(subset(d2,(d2$Comm=='SC'))$bill)
dummy2 <- data.frame("variable" = c("SCR","SSR","bill"), Z = c(scr_sc,ssr_sc, bill_sc))

d2<-subset(d2,d2$Comm=='P2P')
d2$trading<-ordered(d2$trading, levels = c("High", "Normal", "Low"))
d2.m <- melt(select(d2,-X), id.var = c("trading","Comm"))


p_comm <- ggboxplot(d2.m, x= "trading",y="value",fill="trading")
p_comm <- p_comm + geom_hline(data=dummy2, aes(yintercept = Z),colour="red",linetype="longdash")
p_comm <- p_comm + facet_wrap(~variable, scales="free_y",
                              labeller=as_labeller(c("SCR"="SC [%]","SSR"="SS [%]","bill"="Bill [Thousands €]")))
p_comm <- p_comm + stat_compare_means(label.x.npc=0.4,label.y.npc = 0.05)
p_comm <- p_comm + scale_fill_discrete(name =  'Dark2')
p_comm <- p_comm + labs(shape="Type of prosumer:",fill="Type of prosumer:" ,colour="Community:")
p_comm <- p_comm + theme_bw()
p_comm <- p_comm + scale_y_continuous(labels = addUnits)
p_comm <- p_comm + xlab('B) Amount of trading in P2P community')
p_comm <- p_comm + theme(plot.margin = unit(c(0.5,1,0,0), "cm"))
p_comm <- p_comm + theme(legend.position="bottom",legend.text=element_text(size=14),
                         legend.title = element_text(size=16,face="bold"))
p_comm <- p_comm + labs(subtitle = "Community level",face="bold")
p_comm <- p_comm + theme(plot.subtitle = element_text(hjust = 0.5,size=16,face="bold"))
p_comm <- p_comm + theme(axis.text=element_text(size=14),
                         axis.title=element_text(size=14,face="bold"))
p_comm <- p_comm + guides(colour=FALSE, fill=FALSE)

#p_comm


grid.arrange(p_hh,p_comm,nrow=2, ncol=1)



ggsave('../Img/P2P_diff.pdf',plot=grid.arrange(p_hh,p_comm,nrow=2, ncol=1),width=15, height=8.5,
       encoding = "ISOLatin9.enc")

###########################FIGURE AUX###################################
d<-read.table('Optim_trading.csv',sep=',',header=TRUE,stringsAsFactors = FALSE)

d1<-subset(d,d$Comm=='P2P')
p_bill <- ggboxplot(d1, x= "trading",y="bill_hh",fill="trading")+facet_grid(~index)

p_bill <- p_bill + stat_compare_means(label.y=800)

p_bill <- p_bill + scale_colour_manual(values = alpha(c("#F8766D", "#00BFC4"),.3),
                                       labels=c('P2P','Self-consumption'))
p_bill <- p_bill + labs(shape="Type of prosumer:",fill="Type of prosumer:" ,colour="Community:")

p_bill <- p_bill + ylab('Bill\n[€ p.a.]')
p_bill <- p_bill + xlab('P2P trading')
p_bill <- p_bill + theme_bw()
p_bill <- p_bill + theme(plot.margin = unit(c(0.5,1,0,0), "cm"))

p_bill <- p_bill + theme(legend.position="bottom",legend.text=element_text(size=14),
                         legend.title = element_text(size=16,face="bold"))
#p_bill <- p_bill + labs(subtitle = "C) Household level",face="bold")
p_bill <- p_bill + theme(plot.subtitle = element_text(hjust = 0.5,size=16,face="bold"))
p_bill <- p_bill + theme(axis.text=element_text(size=14),
                         axis.title=element_text(size=14,face="bold"))
p_bill <- p_bill + guides(colour=FALSE, fill=FALSE)


p_ssr <- ggboxplot(d1, x= "trading",y="SSR_hh",fill="trading")+facet_grid(~index)

p_ssr <- p_ssr + stat_compare_means()

p_ssr <- p_ssr + scale_colour_manual(values = alpha(c("#F8766D", "#00BFC4"),.3),
                                     labels=c('P2P','Self-consumption'))
p_ssr <- p_ssr + labs(shape="Type of prosumer:",fill="Type of prosumer:" ,colour="Community:")

p_ssr <- p_ssr + ylab('SSR\n[%]')
p_ssr <- p_ssr + xlab('P2P trading')
p_ssr <- p_ssr + theme_bw()
p_ssr <- p_ssr + theme(plot.margin = unit(c(0.5,1,0,0), "cm"))

p_ssr <- p_ssr + theme(legend.position="bottom",legend.text=element_text(size=14),
                       legend.title = element_text(size=16,face="bold"))
#p_ssr <- p_ssr + labs(subtitle = "C) Household level",face="bold")
p_ssr <- p_ssr + theme(plot.subtitle = element_text(hjust = 0.5,size=16,face="bold"))
p_ssr <- p_ssr + theme(axis.text=element_text(size=14),
                       axis.title=element_text(size=14,face="bold"))
p_ssr <- p_ssr + guides(colour=FALSE, fill=FALSE)


p_scr <- ggboxplot(d1, x= "trading",y="SCR_hh",fill="trading")+facet_grid(~index)
p_scr <- p_scr + stat_compare_means()

p_scr <- p_scr + scale_colour_manual(values = alpha(c("#F8766D", "#00BFC4"),.3),
                                     labels=c('P2P','Self-consumption'))
p_scr <- p_scr + labs(shape="Type of prosumer:",fill="Type of prosumer:" ,colour="Community:")

p_scr <- p_scr + ylab('SCR\n[%]')
p_scr <- p_scr + xlab('P2P trading')
p_scr <- p_scr + theme_bw()
p_scr <- p_scr + theme(plot.margin = unit(c(0.5,1,0,0), "cm"))

p_scr <- p_scr + theme(legend.position="bottom",legend.text=element_text(size=14),
                       legend.title = element_text(size=16,face="bold"))
#p_scr <- p_scr + labs(subtitle = "C) Household level",face="bold")
p_scr <- p_scr + theme(plot.subtitle = element_text(hjust = 0.5,size=16,face="bold"))
p_scr <- p_scr + theme(axis.text=element_text(size=14),
                       axis.title=element_text(size=14,face="bold"))
p_scr <- p_scr + guides(colour=FALSE, fill=FALSE)
grid.arrange(p_scr,p_ssr,p_bill,p_bill2,nrow=2, ncol=2)


####################Other#######################
al=0.02
p_peak2 <- p_peak2 + geom_rect(aes(xmin = 0,xmax = 24,
                                   ymin = -Inf, ymax = Inf, fill = 'Monday'), alpha = al)
p_peak2 <- p_peak2 + scale_fill_brewer(palette = 'Pastel1', name = 'Day of the week')
p_peak2 <- p_peak2 + geom_rect(aes(xmin = 24,xmax = 48,
                                   ymin = -Inf, ymax = Inf, fill = 'Tuesday'), alpha = al)
p_peak2 <- p_peak2 + geom_rect(aes(xmin = 48,xmax = 72,
                                   ymin = -Inf, ymax = Inf, fill = 'Wednesday'), alpha = al)
p_peak2 <- p_peak2 + geom_rect(aes(xmin = 72,xmax = 96,
                                   ymin = -Inf, ymax = Inf, fill = 'Thursday'), alpha = al)
p_peak2 <- p_peak2 + geom_rect(aes(xmin = 96,xmax = 120,
                                   ymin = -Inf, ymax = Inf, fill = 'Friday'), alpha = al)
p_peak2 <- p_peak2 + geom_rect(aes(xmin = 120,xmax = 144,
                                   ymin = -Inf, ymax = Inf, fill = 'Saturday'), alpha = al)
p_peak2 <- p_peak2 + geom_rect(aes(xmin = 144,xmax = 168,
                                   ymin = -Inf, ymax = Inf, fill = 'Sunday'), alpha = al)
p_peak2 <- p_peak2 + scale_fill_brewer(palette = 'Pastel1', name = 'Day of the week')
p_peak2 <- p_peak2 +  annotate("rect",xmin = 0,xmax = 24,
                               ymin = -Inf, ymax = Inf, fill = 'red', alpha = .1)
p_peak2 <- p_peak2 + annotate("rect", xmin = 24,xmax = 48,
                              ymin = -Inf, ymax = Inf, fill = 'Tuesday', alpha = .1)
p_peak2 <- p_peak2 + annotate("rect", xmin = 48,xmax = 72,
                              ymin = -Inf, ymax = Inf, fill = 'Wednesday', alpha = .1)
p_peak2 <- p_peak2 + annotate("rect", xmin = 72,xmax = 96,
                              ymin = -Inf, ymax = Inf, fill = 'Thursday', alpha = .1)
p_peak2 <- p_peak2 + annotate("rect", xmin = 96,xmax = 120,
                              ymin = -Inf, ymax = Inf, fill = 'Friday', alpha = .1)
p_peak2 <- p_peak2 + annotate("rect", xmin = 120,xmax = 144,
                              ymin = -Inf, ymax = Inf, fill = 'Saturday', alpha = .1)
p_peak2 <- p_peak2 + annotate("rect", xmin = 144,xmax = 168,
                              ymin = -Inf, ymax = Inf, fill = 'Sunday', alpha = .1)

df_2 <- subset(df,df$index=='No')

p_pi <- ggplot(df_2, aes(y=PI))
p_pi <- p_pi + geom_boxplot(fill="#F8766D")
p_pi <- p_pi + ylab('Participation willingness\n index [p.u.]')
p_pi <- p_pi + xlab('P2P')
p_pi <- p_pi + ylim(0,1)

p_pi <- p_pi + theme_bw()
p_pi <- p_pi + theme(legend.position="bottom",legend.text=element_text(size=14),
                     legend.title = element_text(size=16,face="bold"))
p_pi <- p_pi + labs(subtitle = "E) Participation index",face="bold")
p_pi <- p_pi + theme(plot.subtitle = element_text(hjust = 0.5,size=16,face="bold"))
p_pi <- p_pi + theme(axis.text=element_text(size=14),
                     axis.title=element_text(size=14,face="bold"))

summary(df_2$PI)
