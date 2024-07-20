from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameters
kernels = ['rbf']
gammas = [0.5]
Cs = [0.01, 1, 10]

best_accuracy = 0
best_support_vectors = None

# Train SVM classifiers with different hyperparameters
for kernel in kernels:
    for gamma in gammas:
        for C in Cs:
            clf = SVC(kernel=kernel, gamma=gamma, C=C, decision_function_shape='ovr')
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            support_vectors = clf.n_support_.sum()
            print(f"Kernel: {kernel}, Gamma: {gamma}, C: {C}, Accuracy: {accuracy}, Support Vectors: {support_vectors}")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_support_vectors = support_vectors

print("\nBest classification accuracy:", best_accuracy)
print("Total number of support vectors on test data for best accuracy:", best_support_vectors)
