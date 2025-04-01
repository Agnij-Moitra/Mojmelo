from mojmelo.Adaboost import Adaboost
from mojmelo.DecisionTree import DecisionTree
from mojmelo.KNN import KNN
from mojmelo.LinearRegression import LinearRegression
from mojmelo.LogisticRegression import LogisticRegression
from mojmelo.NaiveBayes import GaussianNB, MultinomialNB
from mojmelo.Perceptron import Perceptron
from mojmelo.PolynomialRegression import PolyRegression
from mojmelo.RandomForest import RandomForest
from mojmelo.SVM import SVM_Primal, SVM_Dual
from mojmelo.utils.utils import CVM
from mojmelo.utils.Matrix import Matrix
from collections import Dict, List
from python import Python
from memory import UnsafePointer
import random


@value
struct ModelResult:
    var model_type: String
    var model_index: Int
    var loss: Float32


struct MSBoost:
    var n_estimators: Int
    var learning_rate: Float32
    var model_names: List[String]
    var models: List[CVM]
    var ensemble: List[Tuple[String, Int]]
    var errors: List[Float32]
    var return_best: Bool
    var custom_metrics: fn(Matrix, Matrix) raises -> Float32
    var is_classifier: Bool

    fn __init__(mut self, 
               n_estimators: Int = 100, 
               learning_rate: Float32 = 0.01,
               is_classifier: Bool = False,
               return_best: Bool = True):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.model_names = List[String]()
        self.models = List[CVM]()
        self.ensemble = List[Tuple[String, Int]]()
        self.errors = List[Float32]()
        self.return_best = return_best
        self.is_classifier = is_classifier
        
        # Register available models
        self.register_models()
    
    fn register_models(mut self):
        # Add model types to the registry
        self.model_names.append("DecisionTree")
        self.model_names.append("LinearRegression")
        self.model_names.append("KNN")
        self.model_names.append("RandomForest")
        self.model_names.append("LogisticRegression")
        self.model_names.append("Perceptron")
        self.model_names.append("PolyRegression")
        self.model_names.append("SVM_Primal")
        self.model_names.append("GaussianNB")
        self.model_names.append("Adaboost")
        
        # Create a default instance of each model type
        var dt = DecisionTree()
        self.models.append(dt)
        
        var lr = LinearRegression()
        self.models.append(lr)
        
        var knn = KNN()
        self.models.append(knn)
        
        var rf = RandomForest()
        self.models.append(rf)
        
        var logr = LogisticRegression()
        self.models.append(logr)
        
        var perceptron = Perceptron()
        self.models.append(perceptron)
        
        var poly = PolyRegression()
        self.models.append(poly)
        
        var svm = SVM_Primal()
        self.models.append(svm)
        
        var gnb = GaussianNB()
        self.models.append(gnb)
        
        var ada = Adaboost()
        self.models.append(ada)
    
    fn default_metric(self, y_true: Matrix, y_pred: Matrix) raises -> Float32:
        """Calculate mean squared error as the default metric."""
        var diff = y_true - y_pred
        var squared = diff.elemwise_multiply(diff)
        return squared.mean()
    
    fn create_model(self, model_type: String) raises -> CVM:
        """Create a new model instance based on the model type."""
        if model_type == "DecisionTree":
            return DecisionTree()
        elif model_type == "LinearRegression":
            return LinearRegression()
        elif model_type == "KNN":
            return KNN()
        elif model_type == "RandomForest":
            return RandomForest()
        elif model_type == "LogisticRegression":
            return LogisticRegression()
        elif model_type == "Perceptron":
            return Perceptron()
        elif model_type == "PolyRegression":
            return PolyRegression()
        elif model_type == "SVM_Primal":
            return SVM_Primal()
        elif model_type == "GaussianNB":
            return GaussianNB()
        elif model_type == "Adaboost":
            return Adaboost()
        else:
            # Default to DecisionTree if model type not recognized
            return DecisionTree()
    
    fn evaluate_model(self, model_type: String, model_idx: Int, X_train: Matrix, y_train: Matrix, X_val: Matrix, y_val: Matrix) raises -> ModelResult:
        """Train a model and evaluate its performance."""
        var model = self.create_model(model_type)
        model.fit(X_train, y_train)
        var y_pred = model.predict(X_val)
        var loss: Float32 = 0.0
        
        # Use custom metric if provided, otherwise use default MSE
        try:
            loss = self.custom_metrics(y_val, y_pred)
        except:
            loss = self.default_metric(y_val, y_pred)
        
        return ModelResult(model_type=model_type, model_index=model_idx, loss=loss)
    
    fn train_test_split(self, X: Matrix, y: Matrix, test_size: Float32 = 0.2) raises -> Tuple[Matrix, Matrix, Matrix, Matrix]:
        """Split the data into training and validation sets."""
        var n_samples = X.height
        var n_test = Int(test_size * n_samples)
        var n_train = n_samples - n_test
        
        # Generate random indices for the split
        var py = Python()
        var indices = py.evaluate("list(range(" + String(n_samples) + "))")
        py.evaluate("import random; random.shuffle(" + indices + ")")
        
        var train_indices = py.evaluate(indices + "[:"+String(n_train)+"]")
        var test_indices = py.evaluate(indices + "["+String(n_train)+":]")
        
        # Create train and test sets
        var X_train = X.get_rows(train_indices)
        var y_train = y.get_rows(train_indices)
        var X_test = X.get_rows(test_indices)
        var y_test = y.get_rows(test_indices)
        
        return (X_train, X_test, y_train, y_test)
    
    fn fit(mut self, X: Matrix, y: Matrix, custom_metrics: fn(Matrix, Matrix) raises -> Float32 = None) raises:
        """Fit the MSBoost model to the data."""
        # Initialize mean prediction
        var y_mean = y.mean()
        var n_samples = X.height
        
        # Store the custom evaluation metric if provided
        if custom_metrics:
            self.custom_metrics = custom_metrics
        
        # Initialize predictions and residuals
        var predictions = Matrix.full(n_samples, 1, y_mean)
        var residuals = y - predictions
        
        for i in range(self.n_estimators):
            print("Training ensemble layer", i+1)
            
            # Split the data
            var X_train: Matrix
            var X_val: Matrix
            var r_train: Matrix
            var r_val: Matrix
            (X_train, X_val, r_train, r_val) = self.train_test_split(X, residuals)
            
            var best_model_type = ""
            var best_model_index = -1
            var best_loss = Float32.max
            
            # Evaluate each model type and find the best performer
            for j in range(len(self.model_names)):
                var model_type = self.model_names[j]
                var result = self.evaluate_model(model_type, j, X_train, r_train, X_val, r_val)
                
                print("  Model:", model_type, "Loss:", result.loss)
                
                if result.loss < best_loss:
                    best_loss = result.loss
                    best_model_type = model_type
                    best_model_index = j
            
            print("  Selected model:", best_model_type)
            
            # Train the best model on the full dataset
            var best_model = self.create_model(best_model_type)
            best_model.fit(X, residuals)
            
            # Update predictions and residuals
            var model_predictions = best_model.predict(X)
            predictions = predictions + model_predictions * self.learning_rate
            residuals = y - predictions
            
            # Calculate error (MSE)
            var current_error = self.default_metric(y, predictions)
            self.errors.append(current_error)
            
            # Save the best model to the ensemble
            self.ensemble.append((best_model_type, i))
            
            # If we've reached perfect prediction, stop early
            if current_error == 0 and self.return_best:
                print("Perfect fit achieved at iteration", i+1)
                break
        
        # Trim ensemble to best performing level if return_best is True
        if self.return_best and len(self.errors) > 1:
            var min_error = Float32.max
            var min_error_idx = 0
            
            for i in range(len(self.errors)):
                if self.errors[i] < min_error:
                    min_error = self.errors[i]
                    min_error_idx = i
            
            print("Best ensemble size:", min_error_idx + 1, "with error:", min_error)
            
            # Trim the ensemble to the best performing subset
            var trimmed_ensemble = List[Tuple[String, Int]]()
            for i in range(min_error_idx + 1):
                trimmed_ensemble.append(self.ensemble[i])
            self.ensemble = trimmed_ensemble
    
    fn predict(self, X: Matrix) raises -> Matrix:
        """Generate predictions with the trained ensemble."""
        var n_samples = X.height
        var predictions = Matrix.full(n_samples, 1, 0.0)
        
        # Sum predictions from all models in the ensemble
        for i in range(len(self.ensemble)):
            var model_type = self.ensemble[i][0]
            var model = self.create_model(model_type)
            model.fit(X, predictions)  # This needs the original training data, which is a limitation
            
            var model_predictions = model.predict(X)
            predictions = predictions + model_predictions * self.learning_rate
        
        # For classification, map values to classes
        if self.is_classifier:
            return predictions.map(fn(x: Float32) -> Float32: return 1.0 if x > 0.5 else 0.0)
        
        return predictions
    
    fn get_errors(self) -> List[Float32]:
        """Return the error history during training."""
        return self.errors
    
    fn get_ensemble(self) -> List[Tuple[String, Int]]:
        """Return the models in the ensemble."""
        return self.ensemble


# Create a classifier version
struct MSBoostClassifier(MSBoost):
    fn __init__(inout self, 
               n_estimators: Int = 100, 
               learning_rate: Float32 = 0.01,
               return_best: Bool = True):
        super().__init__(n_estimators, learning_rate, True, return_best)