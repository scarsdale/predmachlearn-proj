library(caret)
#library(doMC)
#registerDoMC()
set.seed(0451)
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
        list(trainset[,completecols],
             testset[,completecols])
    }
}
workingsets <- mkcommonsubset(gettrainset, gettestset)
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
    #RNGkind("L'Ecuyer-CMRG")
    set.seed(0451)
    #mclapply(algs, mktrain(dat))
    lapply(algs, mktrain(dat))
}
test.models <- function(dat, models) {
    #mclapply(models, mkconfusion(dat))
    lapply(models, mkconfusion(dat))
}
mktimer <- function(fn) {
    function(...) {
        system.time(fn(...))
    }
}
time.training <- function(dat, algs) {
    lapply(algs, mktimer(mktrain(dat)))
}
time.prediction <- function(dat, models) {
    lapply(models, mktimer(mkconfusion(dat)))
}
# selected based on comparison at
# http://www.cs.cornell.edu/~caruana/ctp/ct.papers/caruana.icml06.pdf
# originally tried ada and blackboost also, but they don't seem to work here
candidate.algorithms <- c("parRF",
                          "treebag",
                          "svmLinear",
                          "nnet")
split.data <- function(dat, p) {
    createDataPartition(dat[,isoutcome(dat)], list=F, p=p)    
}
tryout <- function(p) {
    sets <- workingsets()
    trainset <- sets[[1]]
    subtrainidx <- split.data(trainset, p)
    subtrain <- trainset[subtrainidx,]
    subtest <- trainset[-subtrainidx,]
    #algs <- candidate.algorithms
    algs <- c("rf", "svmLinear", "nnet")
    train.times <- time.training(subtrain, algs)
    models <- train.algorithms(subtrain, algs)
    prediction.times <- time.prediction(subtest, models)
    confusions <- test.models(subtest, models)
    list(train.times=train.times,
         models=models,
         prediction.times=prediction.times,
         confusions=confusions)
}
