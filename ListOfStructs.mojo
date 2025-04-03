from utils import Variant
trait Estimator(CollectionElement):
    fn fit(self): 
        ...
    fn predict(self): 
        ...

@value
struct KNN(Estimator):
   
    fn __init__(out self):
        print("KNN init")
    
    fn fit(self):
        print("KNN fit")

    fn predict(self):
        print("KNN predict")
        
@value
struct SVM(Estimator):
    
    fn __init__(out self):
        print("SVM Init")
    
    fn fit(self):
        print("SVM Fit")
    
    fn predict(self):
        print("SVM predict")

fn main():
    var estimators = List[Variant[KNN, SVM]]()
    estimators.append(KNN())
    estimators.append(SVM())
    # For iterating:
    for estimator in estimators:
        if estimator[].isa[KNN]():
            estimator[][KNN].fit()
            estimator[][KNN].predict()
        else:
            estimator[][SVM].fit()
            estimator[][SVM].predict()
