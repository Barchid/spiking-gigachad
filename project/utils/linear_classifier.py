from sklearn.svm import SVC
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from snntorch import spikegen


def Total(encoder, dataset, dataset_name, is_ann):
    with torch.no_grad():
        data_total, target_total = [], []
        i = 0
        for data, target in dataset:
            print(i, "/", len(dataset))

            if "dvs" in dataset_name:
                data = data.to(torch.float)
            else:
                data = data / 255
                data = spikegen.rate(data, 15)

            if is_ann:
                data = data.sum(0) / 15.0

            data = data.to(device)

            feats = encoder(data).cpu()
            data_total.append(feats)
            target_total.append(target)
            i += 1
    return torch.cat(data_total), torch.cat(target_total)


def classification(encoder, train_set, val_set, dataset, is_ann):
    print("Total features - TRAIN")
    train_data, train_target = Total(encoder, train_set, dataset, is_ann)

    print("Total features - TEST")
    test_data, test_target = Total(encoder, val_set, dataset, is_ann)
    print("Shape of test data:", test_data.shape)
    torch.save(test_data, f"experiments/features/{dataset}_{is_ann}.pt")

    print("PCA")
    pca = PCA(n_components=200).fit(train_data, train_target)
    train_data, test_data = pca.transform(train_data), pca.transform(test_data)

    print("SVM")
    target = SVC(C=2.4).fit(train_data, train_target).predict(test_data)
    accuracy = (torch.tensor(target) == test_target).sum() / len(test_target)
    mess = f"Final Accuracy: {accuracy * 100 :.2f}%\n"
    print(mess)
    with open(f"report_{dataset}_{is_ann}.txt", "a") as fp:
        fp.write(f"==> DATA={dataset} IS_ANN={is_ann} ==> ")
        fp.write(mess)
        fp.flush()
    return accuracy
