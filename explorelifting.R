library(caret)
library(doMC)
registerDoMC()
download.server <- "https://d396qusza40orc.cloudfront.net"
download.basedir <- "predmachlearn"
mkurl <- function(file) {
    paste(download.server, download.basedir, file, sep="/")
}
mkgetdataset <- function(filename) {
    function() {
        if (!file.exists(filename)) {
            download.file(mkurl(filename), destfile=filename, method="curl")
        }
        read.csv(filename)
    }
}
mkprocdataset <- function(getfn) {
    function () {
        dat <- getfn()
        # remove columns that are not predictors
        rownames(dat) <- dat$X
        dat$X <- NULL
        dat$user_name <- NULL
        # the assignment appears to require ignoring these
        dat$raw_timestamp_part_1 <- NULL
        dat$raw_timestamp_part_2 <- NULL
        dat$cvtd_timestamp <- NULL
        dat$new_window <- NULL
        dat$num_window <- NULL
        # 
        dat
    }
}
gettrainset <- mkprocdataset(mkgetdataset("pml-training.csv"))
gettestset <- mkprocdataset(mkgetdataset("pml-testing.csv"))
sapplycols <- function(dat, fn) {
    sapply(colnames(dat), function(colname) { fn(dat[,colname]) })
}
isnumericwithNA <- function(v) {
    is.numeric(v) && any(is.na(v))
}
isnumericwithoutNA <- function(v) {
    is.numeric(v) && !any(is.na(v))
}
isallNA <- function(v) {
    all(is.na(v))
}
isanyNA <- function(v) {
    any(is.na(v))
}
iscomplete <- function(v) {
    !any(is.na(v))
}
whichallNA <- function(dat) {
    sapplycols(dat, isallNA)
}
completecols <- function(dat) {
    sapplycols(dat, iscomplete)
}
mkcommonsubset <- function(gettrain, gettest) {
    function () {
        trainset <- gettrain()
        testset <- gettest()        
        completecols <- intersect(which(completecols(trainset)),
                                  which(completecols(testset)))
        list(training=trainset[,completecols],
             testing=testset[,completecols])
    }
}
workingsets <- mkcommonsubset(gettrainset, gettestset)
mkpreproc <- function(dat, ...) {
    preproc <- preProcess(dat[,ispredictor(dat)], ...)
    function(dat) {
        ndat <- dat
        ndat[,ispredictor(dat)] <- predict(preproc, dat[,ispredictor(dat)])
        ndat
    }
}
isoutcome <- function(dat) {
    colnames(dat) %in% c("classe", "problem_id")
}
ispredictor <- function(dat) {
    !isoutcome(dat)
}
correlation.matrix <- function(dat) {
    # from module 016preProcessingPCA slide 2
    M <- abs(cor(dat[,ispredictor(dat)]))
    diag(M) <- 0
    M
}
do.rfcv <- function(dat) {
    rfcv(dat[,ispredictor(dat)], dat[,isoutcome(dat)])
}
mktrain <- function(dat) {
    function(algname) {
        train(dat[,ispredictor(dat)], dat[,isoutcome(dat)], method=algname)
    }
}
mkconfusion <- function(dat) {
    function(model) {
        predictions <- predict(model, dat)
        confusionMatrix(predictions, dat[,isoutcome(dat)])
    }
}
train.algorithms <- function(dat, algs) {
    set.seed(0451)
    lapply(algs, mktrain(dat))
}
test.models <- function(dat, models) {
    lapply(models, mkconfusion(dat))
}
mktimer <- function(fn) {
    function(...) {
        t <- system.time(res <- fn(...))
        list(t=t, res=res)
    }
}
timer.getresults <- function(l) {
    lapply(l, function(ll) { ll$res })
}
timer.gettimes <- function(l) {
    lapply(l, function(ll) { ll$t })
}
time.training <- function(dat, algs) {
    res <- lapply(algs, mktimer(mktrain(dat)))
    list(models=timer.getresults(res),
         times=timer.gettimes(res))
}
time.prediction <- function(dat, models) {
    res <- lapply(models, mktimer(mkconfusion(dat)))
    list(confusions=timer.getresults(res),
         times=timer.gettimes(res))
}
# selected based on comparison at
# http://www.cs.cornell.edu/~caruana/ctp/ct.papers/caruana.icml06.pdf
# originally tried ada, treebag, and blackboost also,
# but couldn't get them to work
candidate.algorithms <- c("parRF",
                          "nb",
                          "nnet")
split.data <- function(dat, p) {
    createDataPartition(dat[,isoutcome(dat)], list=F, p=p)    
}
getsplit.trainset <- function(p) {
    sets <- workingsets()
    trainset <- sets[[1]]
    subtrainidx <- split.data(trainset, p)
    osubtrain <- trainset[subtrainidx,]
    preproc <- mkpreproc(osubtrain, method=c("center", "scale"))
    subtrain <- preproc(osubtrain)
    subtest <- preproc(trainset[-subtrainidx,])
    list(training=subtrain,
         testing=subtest)
}
try.models <- function(p) {
    subsets <- getsplit.trainset(p)
    algs <- candidate.algorithms
    train.res <- time.training(subsets$training, algs)
    prediction.res <- time.prediction(subsets$testing, train.res$models)
    list(train.times=train.res$times,
         models=train.res$models,
         prediction.times=prediction.res$times,
         confusions=prediction.res$confusions)
}
elapsedtime <- function(t) {
    t[3]
}
analyze.res <- function(res) {
    data.frame(time.train=sapply(res$train.times, elapsedtime),
               time.predict=sapply(res$prediction.times, elapsedtime),
               train.accuracy=sapply(res$models, function(m) {
                   max(m$results$Accuracy) }),
               test.accuracy=sapply(res$confusions, function(cm) {
                   cm$overall["Accuracy"] }),
               row.names=sapply(res$models, function(m) { m$method }))
}
