# senior-design-ml

This program creates a classifier neural network model. The program currently outputs two arrays. The first array describes the predicted classes. The second array is two dimensional and describes the prediction confidences for each class. 

This model has 3 layers with 53, 24, and 6 nodes, respectively. 

To run this model locally, create a virtual environment and execute the file using the command python csi_model6.py’.

This message prints every time the program executes:

tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2021-11-06 12:59:55.263105: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)

This is to be expected. This message describes some optimization capabilities of Tensorflow. If this creates an issue, I can find a way to suppress the message. For more information, check this out: https://stackoverflow.com/questions/65298241/what-does-this-tensorflow-message-mean-any-side-effect-was-the-installation-su

