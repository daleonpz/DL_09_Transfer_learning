import numpy as np
import torch
import torch.utils.data
import torchvision
import sklearn.metrics
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from torch import nn


class KnnConvnet:
    def __init__(self, model, device="cpu", distance="cosine"):
        super(KnnConvnet, self).__init__()
        self.device = device
        self.model = model.to(device)
        self.model.eval()
        self.embeds_train = None
        self.lab_train = None
        self.lab_test = None
        self.embeds_test = None
        self.distance = distance
        self.knnClassifier = None

    def get_features(self, mode="train"):
        """Returns the embeddings of the train or test set.
        Args:
            mode (str, optional): "train" or "test". Defaults to 'train'.
        """
        assert self.embeds_train is not None, "Training embedding are not computed yet."
        assert self.embeds_test is not None, "Test embedding are not computed yet."
        ### START CODE HERE ### (approx. 4 lines)
        if mode == "train":
            return self.embeds_train, self.lab_train
        if mode == "test":
            return self.embeds_test, self.lab_test
        ### END CODE HERE ###

    def set_features(self, embeds, labels, mode="train"):
        """Sets the train or test embeddings and their labels."""
        ### START CODE HERE ### (approx. 6 lines)
        if mode == "train":
            self.embeds_train = embeds
            self.lab_train = labels
        if mode == "test":
            self.embeds_test = embeds
            self.lab_test = labels
        ### END CODE HERE ###

    @torch.no_grad()
    def extract_features(self, loader):
        """Infers features from the provided image loader.
        Args:
            loader: train or test loader
        Returns: 3 tensors of all: features, labels
        """
        feature_list = []
        label_list = []
        ### START CODE HERE ### (approx. 4 lines)
        for inputs, labels in loader:
            # to(device) needed so the model and input are on the same 'device'
            features = self.model(inputs.to(self.device))  
            feature_list.append(features)
            label_list.append(labels)
        ### END CODE HERE ###
        h_total = torch.cat(feature_list)
        label_total = torch.cat(label_list)
        return h_total, label_total

    @torch.no_grad()
    def fit(self, features, labels, k):
        """Fits the provided features to create a KNN classifer.
        Args:
            features: [... , dataset_size, feat_dim]
            labels: [... , dataset_size]
            k: number of nearest neighbours for majority voting
        Returns: train accuracy, or train and test acc
        """
        ### START CODE HERE ### (approx. 2 lines)
        self.knnClassifier = KNeighborsClassifier(n_neighbors=k, metric=self.distance)
        # DataConversionWarning: A column-vector y was passed when a 1d array was expected. 
        # Please change the shape of y to (n_samples,), for example using ravel()
        # ravel will convert that array shape to (n, )
        self.knnClassifier.fit( features.cpu().numpy(), labels.ravel())  
        ### END CODE HERE ###

    def predict(self, features, labels):
        """Uses the features to compute the accuracy of the classifier (self.cls object)."""
        ### START CODE HERE ### (approx. 2 lines)
        prediction = self.knnClassifier.predict(features.cpu().numpy())
        prediction = torch.tensor(prediction)
        acc = (prediction == torch.reshape(labels,(-1,) )).sum().item() / labels.shape[0] 
        acc *= 100
        ### END CODE HERE ###
        return acc

    @torch.no_grad()
    def execute(self, train_loader, test_loader=None, k=10):
        if self.embeds_train is None:
            embeds_train, lab_train = self.extract_features(train_loader)
            self.set_features(embeds_train, lab_train, mode="train")

        self.fit(self.embeds_train, self.lab_train, k)
        train_acc = self.predict(self.embeds_train, self.lab_train)


        if test_loader is not None:
            if self.embeds_test is None:
                embeds_test, lab_test = self.extract_features(test_loader)
                self.set_features(embeds_test, lab_test, mode="test")

            test_acc = self.predict(self.embeds_test, self.lab_test)
            return train_acc, test_acc

        return train_acc


def test_knn(train_ds_plain, val_ds):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    d1 = torch.utils.data.Subset(train_ds_plain, list(range(300)))
    d2 = torch.utils.data.Subset(val_ds, list(range(100)))

    train_loader = torch.utils.data.DataLoader(
        dataset=d1, batch_size=32, shuffle=False, drop_last=False
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=d2, batch_size=32, shuffle=False, drop_last=False
    )

    model = torchvision.models.resnet18(pretrained=True)
    knn_cls = KnnConvnet(model, device=device)
    train_acc, test_acc = knn_cls.execute(train_loader, test_loader, k=10)
    print(f"train acc: {train_acc:.2f}%, test acc: {test_acc:.2f}%")
    return train_acc, test_acc
