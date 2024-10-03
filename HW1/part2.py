import os
import csv
import cv2
import click
import random
import numpy as np
import matplotlib.pyplot as plt
from part2 import *
from enum import Enum
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, LeaveOneOut


class MyLogisticRegression:
    def __init__(self, weights_init='random', add_bias=True, learning_rate=1e-5, 
        num_iterations=1_000, verbose=False, max_error=1e-5):
        ''' Linear regression model using gradient descent 

        # Arguments
            weights_init: str
                weights initialization option ['random', 'zeros']
            add_bias: bool
                whether to add bias term 
            learning_rate: float
                learning rate value for gradient descent
            num_iterations: int 
                maximum number of iterations in gradient descent
            verbose: bool
                enabling verbose output
            max_error: float
                error tolerance term, after reaching which we stop gradient descent iterations
        '''

        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.weights_init = weights_init
        self.add_bias = add_bias
        self.verbose = verbose
        self.max_error = max_error

        self.error_history : list[float] = []
    
    def initialize_weights(self, n_features):
        ''' weights initialization function '''
        if self.weights_init == 'random':
            ################

            # YOUR CODE HERE
            weights = np.random.randn(n_features)

            ################
        elif self.weights_init == 'zeros':
            ################

            # YOUR CODE HERE
            weights = np.zeros(n_features)

            ################
        else:
            raise NotImplementedError
        return weights

    @staticmethod
    def cost(target, pred) -> float:
        m = len(target)
        epsilon = 1e-5
        loss = (-1/m) * (target.T @ np.log(pred + epsilon) - (1 - target).T @ np.log(1 - pred + epsilon))

        # loss = np.mean((target - pred) ** 2)
        return loss

    def fit(self, x, y) -> None:
        if self.add_bias:
            ################

            # YOUR CODE HERE
            x = np.hstack([np.ones((x.shape[0], 1)), x])

            ################

        self.weights = self.initialize_weights(x.shape[1])

        y = y.flatten()

        m = len(y)

        for i in range(self.num_iterations):
            y_pred = self.forward(x)

            gradient = (1 / m) * (x.T @ (y_pred - y))
            self.weights -= self.learning_rate * gradient

            new_loss : float = MyLogisticRegression.cost(y, y_pred)
            self.error_history.append(new_loss)

            # if i > 0 and (abs(new_loss - self.error_history[i-1]) < self.max_error):
            #     if self.verbose:
            #         print(f"Converged after {i+1} iterations.")
            #     break
    
    def predict(self, x):
        if self.add_bias and x.shape[1] < self.weights.shape[0]:
            x = np.hstack([np.ones((x.shape[0], 1)), x])

        return (self.forward(x) >= 0.5).astype(int)

    def forward(self, x):
        return MyLogisticRegression.sigmoid(x @ self.weights)

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    

class ImageProcessor:
    def __init__(self, vectorize_method: str, intermediate_image_size: tuple[int]) -> None:
        self.vectorize_method = vectorize_method
        self.img_size = intermediate_image_size

        self.tr_to_grayscale = lambda img : cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        self.tr_rescale = lambda img : cv2.resize(img, self.img_size)
        self.tr_to_1d = lambda img : img.reshape(-1, 1)
        self.tr_flatten = lambda img : img.flatten()
        self.tr_to_float = lambda img : img.astype(np.float32)

    @staticmethod
    def rotate_image(image):
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        angle = 45

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
        return rotated_image
    
    @staticmethod
    def translate_image(image):
        tx, ty = random.randrange(-50, 50), random.randrange(-50, 50)
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])

        (h, w) = image.shape[:2]
        shifted_image = cv2.warpAffine(image, translation_matrix, (w, h))
        return shifted_image
    
    @staticmethod
    def mirror_image(image):
        flipped_image = cv2.flip(image, 1)
        return flipped_image
    
    @staticmethod
    def augment_images(images, labels: list[int]):
        aug_tr = [ImageProcessor.rotate_image, ImageProcessor.translate_image, ImageProcessor.mirror_image]
        augmented = []
        augmented_labels = []

        for i, img in enumerate(images):
            tr_ind = random.randint(0, 2)
            augmented.append(aug_tr[tr_ind](img))
            augmented_labels.append(labels[i])

        return augmented, augmented_labels

    def transform_images(self, images: np.ndarray, labels: list[int], augment: bool) -> np.ndarray:
        default_tr = [
                self.tr_rescale,
                self.tr_to_1d,
                self.tr_flatten,
                self.tr_to_float
            ]

        grayscale_images = [self.tr_to_grayscale(img) for img in images]

        if augment:
            augmented_img, augmented_lbl = ImageProcessor.augment_images(grayscale_images, labels)
            grayscale_images += augmented_img
            new_labels = labels + augmented_lbl

        new_images = np.ndarray((len(grayscale_images), self.img_size[0] * self.img_size[1]))

        for i, img in enumerate(grayscale_images):
            new_image = img
            for tr in default_tr:
                new_image = tr(new_image)
            new_images[i] = new_image

        return (new_images, new_labels) if augment else new_images
    
    def compute(self, images: np.ndarray, labels: list[int], max_components: int) -> None:
        transformed_images, transformed_labels = self.transform_images(images, labels, True)

        self.mean, self.eigenvectors = cv2.PCACompute(transformed_images, mean=None, maxComponents=max_components)
        self.num_components = self.eigenvectors.shape[0]

        return transformed_images, transformed_labels

    def vectorize(self, images: np.ndarray, transform: bool=False):
        ''' Vectorizes images into a matrix of size (N, D), where N is the number of images, and D is the dimensionality of the image.'''

        if transform:
            transformed_images = self.transform_images(images, None, False)
        else:
            transformed_images = images

        X = np.ndarray((len(transformed_images), self.num_components))
        for i in range(len(transformed_images)):
            X[i] = cv2.PCAProject(transformed_images[i].reshape(1, -1), self.mean, self.eigenvectors)

        return X


def load_data(image_folder: str, label_file: str):
    ''' Loads images and labels from the specified folder and file.'''
    # load labels file
    labelsdict = dict()
    with open(label_file) as labelscsv:
        labelsreader = csv.reader(labelscsv, delimiter='|')
        for line in labelsreader:
            labelsdict[line[0]] = line[-1]
    labelsdict.pop("image_name")

    # load corresponding images
    filenames = os.listdir(image_folder)
    
    imagesdict = dict()
    for filename in filenames:
        img = cv2.imread(os.path.join(image_folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imagesdict[filename] = img

    labels = []
    images = []
    random.shuffle(filenames)
    for filename in filenames:
        if filename in labelsdict and filename in imagesdict:
            labels.append(labelsdict[filename])
            images.append(imagesdict[filename])

    return images, labels


def label_to_index(label: str) -> int:
    return 0 if label == "human" else 1


def index_to_label(index: int) -> str:
    return "human" if index == 0 else "animal"


def vectorize_images(images: np.ndarray, new_image_size: tuple[int], max_components: int):
    ''' Vectorizes images into a matrix of size (N, D), where N is the number of images, and D is the dimensionality of the image.'''
    pass


def validation_split(X: np.ndarray, y: np.ndarray, test_percent: float):
    ''' Splits data into train and test.'''
    num_for_train = int(len(X) * (1 - test_percent))
    X_train = X[:num_for_train]
    X_test = X[num_for_train:]
    y_train = y[:num_for_train]
    y_test = y[num_for_train:]
    return X_train, X_test, y_train, y_test


class ModelType(Enum):
    Undefined = 0
    LogisticRegression = 1
    KNN = 2
    DecisionTree = 3

    @staticmethod
    def from_string(type_name: str):
        if type_name == "LogisticRegression":
            return ModelType.LogisticRegression
        elif type_name == "KNN":
            return ModelType.KNN
        elif type_name == "DecisionTree":
            return ModelType.DecisionTree
        else:
            return ModelType.Undefined


def create_model(model_name: ModelType):
    ''' Creates a model of the specified name. 
    1. Use your LinearRegression implementation,
    2. TODO
    3. 
    Args:
        model_name (str): Name of the model to use.
    Returns:
        model (object): Model of the specified name.
    '''

    if model_name == ModelType.LogisticRegression:
        model = MyLogisticRegression(learning_rate=0.00001, num_iterations=1_000_000, add_bias=True, weights_init='random')
    elif model_name == ModelType.KNN:
        model = KNeighborsClassifier(n_neighbors=3)
    elif model_name == ModelType.DecisionTree:
        model = DecisionTreeClassifier(random_state=42)
    else:
        raise Exception("Unkown model type")

    return model


@click.command()
@click.option("--image_folder", type=str, help="Path to the folder containing images")
@click.option("--label_file", type=str, help="Path to the file containing labels")
@click.option("--model_name", type=str, help="Name of the model to use")
@click.option("--test_size", type=float, default=0.2, help="Size of the test split")
def main(image_folder: str, label_file: str, model_name: str, test_size: float):

    # Create dataset of image <-> label pairs
    images, labels = load_data(image_folder, label_file)

    labels = [label_to_index(lbl) for lbl in labels]

    # augment + embedding
    img_size = (64, 64)
    image_processor = ImageProcessor("PCA", img_size)
    tr_images, tr_labels = image_processor.compute(images, labels, 128)

    # shuffle
    temp = list(zip(tr_images, tr_labels))
    random.shuffle(temp)
    tr_images, tr_labels = zip(*temp)

    # Yes, ImageProcessor was created with "PCA" parameter, but now we define bool whether to use PCA, don't ask please ¯\_(ツ)_/¯
    USE_PCA = True

    if USE_PCA:
        X = image_processor.vectorize(tr_images, False)
    else:
        X = np.ndarray((len(tr_images), img_size[0] * img_size[1]))
        for i in range(len(tr_images)):
            X[i] = tr_images[i].reshape(1, -1)


    X_train, X_test, y_train, y_test = validation_split(X, tr_labels, 0.2)
    print(f"train size: {X_train.shape}")
    print(f"test size: {X_test.shape}")

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # create model
    model = create_model(ModelType.from_string(model_name))

    # Train model using different validation strategies (refere to https://scikit-learn.org/stable/modules/cross_validation.html)
    # 1. Train, validation, test splits: so you need to split train into train and validation 
    # 2. K-fold cross-validation: apply K-fold cross-validation on train data
    # 3. Leave-one-out cross-validation: apply Leave-one-out cross-validation on train data

    # Make a prediction on test data
    y_pred = model.predict(X_test) # in the simpliest validation strategy, in K-fold you will have multiple predictions

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Make error analysis
    # 1. Plot the first 10 test images, and on each image plot the corresponding prediction
    # 2. Plot the confusion matrix


if __name__ == "__main__":
    main()
