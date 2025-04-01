from utils import Variant
trait Estimator(CollectionElement):
    fn fit(self): 
        ...
    fn predict(self): 
        ...

@value
struct KNN(Estimator):
   
    fn __init__(out self):
        print("init")
    
    fn fit(self):
        print("fit")

    fn predict(self):
        print("predict")
        
@value
struct SVM(Estimator):
    
    fn __init__(out self):
        print("init")
    
    fn fit(self):
        print("Fit")
    
    fn predict(self):
        print("predict")

fn main():
    var estimators = List[Variant[KNN, SVM]]()
    estimators.append(KNN())
    estimators.append(SVM())
    # For iterating:
    for estimator in estimators:
        if estimator[].isa[KNN]():
            estimator[][KNN].fit()
        else:
            estimator[][SVM].fit()