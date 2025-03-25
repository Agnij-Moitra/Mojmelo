# from mojmelo.utils.Matrix import Matrix
# from mojmelo.utils.utils import CVM, CVP, sigmoid, mse_loss
# from mojmelo.DecisionTree import DecisionTree
# from memory import UnsafePointer
# from collections import List
# from python import PythonObject
# import math
# import random

# # Define a trait for estimators
# trait EstimatorTrait:
#     fn fit(self, X: Matrix, y: Matrix, y_po: PythonObject) raises:
#         pass

#     fn predict(self, X: Matrix) raises -> Matrix:
#         pass

# # GB with GOSS, CatBoost-like transformer, adaptive learning rate, and flexible base estimator
# struct Custom_GB(CVM):
#     var n_trees: Int
#     var learning_rate: Float32
#     var adaptive_lr: Bool
#     var goss_a: Float32
#     var goss_b: Float32
#     var task: String  # "classification" or "regression"
#     var base_estimators: UnsafePointer[EstimatorTrait]
#     var initial_prediction: Float32

#     fn __init__(inout self, n_trees: Int = 100, learning_rate: Float32 = 0.1, adaptive_lr: Bool = True, 
#                 goss_a: Float32 = 0.2, goss_b: Float32 = 0.1, task: String = "classification", 
#                 base_estimator: Optional[EstimatorTrait] = None) raises:
#         self.n_trees = n_trees
#         self.learning_rate = learning_rate
#         self.adaptive_lr = adaptive_lr
#         self.goss_a = goss_a
#         self.goss_b = goss_b
#         self.task = task.lower()
#         if self.task not_in ["classification", "regression"]:
#             raise "Task must be 'classification' or 'regression'"
#         self.base_estimators = UnsafePointer[EstimatorTrait].alloc(n_trees)
#         for i in range(n_trees):
#             var estimator: EstimatorTrait
#             if base_estimator.is_some():
#                 estimator = base_estimator.value()
#             else:
#                 estimator = DecisionTree()
#             UnsafePointer.address_of(self.base_estimators[i]).init_pointee_move(estimator)
#         self.initial_prediction = 0.0

#     fn fit(inout self, X: Matrix, y: Matrix) raises:
#         var transformed_y = self._transform_labels(y)

#         # Set initial prediction based on task
#         var y_mean = transformed_y.mean()
#         if self.task == "classification":
#             self.initial_prediction = math.log(y_mean / (1.0 - y_mean + 1e-6))  # Log-odds for classification
#         else:  # regression
#             self.initial_prediction = y_mean  # Mean for regression

#         var current_prediction = Matrix.full(X.height, 1, self.initial_prediction)
#         var y_po = Matrix.to_numpy(y)

#         for i in range(self.n_trees):
#             var gradients = self._compute_gradients(transformed_y, current_prediction)
#             var hessians = self._compute_hessians(current_prediction)
#             var sampled_indices = self._goss_sampling(gradients, hessians)
#             var sampled_X = X[sampled_indices]
#             var sampled_y = gradients[sampled_indices]
#             self.base_estimators[i].fit(sampled_X, sampled_y, y_po)
#             var tree_prediction = self.base_estimators[i].predict(X)
#             current_prediction += self.learning_rate * tree_prediction
#             if self.adaptive_lr:
#                 self.learning_rate = self._adjust_learning_rate(i)

#     fn predict(self, X: Matrix) raises -> Matrix:
#         var prediction = Matrix.full(X.height, 1, self.initial_prediction)
#         for i in range(self.n_trees):
#             prediction += self.learning_rate * self.base_estimators[i].predict(X)
#         if self.task == "classification":
#             return sigmoid(prediction)  # Probabilities for classification
#         return prediction  # Raw values for regression

#     fn _transform_labels(self, y: Matrix) raises -> Matrix:
#         # Optional transformation; disable for regression if not needed
#         var transformed_y = Matrix(y.height, 1)
#         if self.task == "classification":
#             var mean_target = y.mean()
#             var smoothing = 1.0
#             for i in range(y.height):
#                 transformed_y[i, 0] = (y[i, 0] + smoothing * mean_target) / (1.0 + smoothing)
#         else:  # regression
#             for i in range(y.height):
#                 transformed_y[i, 0] = y[i, 0]  # No transformation
#         return transformed_y

#     fn _compute_gradients(self, y: Matrix, prediction: Matrix) raises -> Matrix:
#         var gradients = Matrix(y.height, 1)
#         if self.task == "classification":
#             var pred_sigmoid = sigmoid(prediction)
#             for i in range(y.height):
#                 gradients[i, 0] = y[i, 0] - pred_sigmoid[i, 0]  # Logistic loss gradient
#         else:  # regression
#             for i in range(y.height):
#                 gradients[i, 0] = y[i, 0] - prediction[i, 0]  # Squared error gradient
#         return gradients

#     fn _compute_hessians(self, prediction: Matrix) raises -> Matrix:
#         var hessians = Matrix(prediction.height, 1)
#         if self.task == "classification":
#             var pred_sigmoid = sigmoid(prediction)
#             for i in range(prediction.height):
#                 hessians[i, 0] = pred_sigmoid[i, 0] * (1.0 - pred_sigmoid[i, 0])  # Logistic loss Hessian
#         else:  # regression
#             for i in range(prediction.height):
#                 hessians[i, 0] = 1.0  # Squared error Hessian (constant)
#         return hessians

#     fn _goss_sampling(self, gradients: Matrix, hessians: Matrix) -> List[Int]:
#         var n_samples = gradients.height
#         var n_large = Int(self.goss_a * n_samples)
#         var n_small = Int(self.goss_b * (n_samples - n_large))
#         var grad_abs = List[Tuple[Float32, Int]]()
#         for i in range(n_samples):
#             grad_abs.append((math.abs(gradients[i, 0]), i))
#         grad_abs.sort()
#         var sampled_indices = List[Int]()
#         for i in range(n_samples - n_large, n_samples):
#             sampled_indices.append(grad_abs[i][1])
#         var small_indices = List[Int]()
#         for i in range(n_samples - n_large):
#             small_indices.append(grad_abs[i][1])
#         random.shuffle(small_indices)
#         for i in range(min(n_small, len(small_indices))):
#             sampled_indices.append(small_indices[i])
#         return sampled_indices

#     fn _adjust_learning_rate(self, iteration: Int) -> Float32:
#         return self.learning_rate / (1.0 + 0.05 * iteration)

#     fn __del__(owned self):
#         for i in range(self.n_trees):
#             (self.base_estimators + i).destroy_pointee()
#         self.base_estimators.free()

# # # Example usage
# # fn main() raises:
# #     # Classification example
# #     var X_class = Matrix.rand(100, 5)
# #     var y_class = Matrix.rand(100, 1)
# #     for i in range(y_class.height):
# #         y_class[i, 0] = 1.0 if y_class[i, 0] > 0.5 else 0.0
# #     var model_class = GBDT_GOSS_CatBoost_XGBoost(n_trees=10, task="classification")
# #     model_class.fit(X_class, y_class)
# #     var pred_class = model_class.predict(X_class)
# #     print("Classification predictions:", pred_class.height, pred_class.width)

# #     # Regression example
# #     var X_reg = Matrix.rand(100, 5)
# #     var y_reg = Matrix.rand(100, 1)  # Continuous values
# #     var model_reg = GBDT_GOSS_CatBoost_XGBoost(n_trees=10, task="regression")
# #     model_reg.fit(X_reg, y_reg)
# #     var pred_reg = model_reg.predict(X_reg)
# #     print("Regression predictions:", pred_reg.height, pred_reg.width)