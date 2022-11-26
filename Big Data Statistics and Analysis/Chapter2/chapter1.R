library(utils)
library(plyr)
library(ggplot2)
LoanStats3c=read.csv("2007-2015loan.csv", skip = 1, header = T, sep = ",", nrow = 1000)
data0<-na.omit(LoanStats3c[,"loan_amnt"])
N<-length(data0)
N

data1<-cut(data0,breaks=c(min(data0)-1,1000+1700*(1:19),max(data0)))
data1<-data.frame(data1)
dataf<-ddply(data1,"data1",summarise,n=length(data1))
dataf$组号<-c(1:20)
ggplot(dataf,aes(x=组号,y=n))+geom_bar(stat="identity")+geom_text(aes(label=n),color="black",vjust=-0.3)+labs(title="贷款额度",xlab="小组编号",ylab="频数")

data2<-cut(data0,breaks=c(min(data0)-1,1000+1700*(1:15),29900,33300,max(data0)))
data2<-data.frame(data2)
dataf<-ddply(data2,"data2",summarise,n=length(data2))
dataf组号<-c(1:18)
write.csv(dataf,"dataf.csv")
ggplot(dataf,aes(x=组号，y=n))+geom_bar(stat="identity")+geom_text(aes(label=n),color="black",vjust=-0.3)+labs(title="贷款额度",xlab="小组编号",ylab="频数")

PD<-table(data2)/N
PD
samp=c(100,1000,5000,10000)

n<-length(samp)
fun1<-function(i){
 #set.seed(1)
 p<-sample(data0,i)
 p<-c(p,matrix(NA,1,max(samp)-length(p)))
 return(p)
}

samp<-as.matrix(samp)
ma<-apply(samp,1,fun1)
write.csv(ma,"ma.csv")

fun2<-function(datasam1){
datasam11<-cut(na.omit(datasam1),breaks=c(min(data0)-1,1000+1700*(1:15),29900,33300,max(data0)))
 PS<-table(datasam11)/length(na.omit(datasam1))+0.0000000001
 J<-sum((PS-PD)*(log(PS/PD)))
 q<-exp(-J)
 return(q)
}

Q1<-apply(ma,2,fun2)
write.csv(Q1,"Q1.csv")

x=seq(6,15,by=0.1)
y=2^(x)
plot(x,y,ylab="样本容量",xlab="自变量",main="选取样本容量",col="blue")
head(y,20)

samp=round(y)[-c(1:10)]
n<-length(samp)
n

fun1<-function(i){
 #set.seed(1)
 p<-sample(data0,i)
 p<-c(p,matrix(NA,1,samp[n]-length(p)))
 return(p)
}

samp<-as.matrix(samp)
ma<-apply(samp,1,fun1)

fun2<-function(datasam1){
 datasam11<-cut(na.omit(datasam1),breaks=c(min(data0)-1,1000+1700*(1:15),29900,33300,max(data0)))
 PS<-table(datasam11)/length(na.omit(datasam1))+0.0000000001
 J<-sum((PS-PD)*(log(PS/PD)))
 q<-exp(-J)
 return(q)
}

Q1<-apply(ma,2,fun2)
plot(samp,Q1,xlab="样本质量",ylab="样本容量",main="选取样本容量")
input<-data.frame(samp,Q1)
names(input)<-c("样本容量","样本质量")
write.csv(input,"input.csv")
which(input$样本质量>0.99)
input[34,1]
