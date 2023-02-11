from sklearn.svm import SVC
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from spikingjelly.clock_driven import surrogate, neuron, functional, layer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from snntorch import spikegen


def Total(encoder, dataset, dataset_name, is_ann):
    with torch.no_grad():
        data_total, target_total = [], []
        i = 0
        for data, target in dataset:
            print(i, "/", len(dataset))

            if "dvs" in dataset_name:
                data = data.to(torch.float)  # BTCHW
            else:
                data = data / 255  # normalize
                data = spikegen.rate(data, 15)  # TBCHW
                data = data.permute(1, 0, 2, 3, 4) # BTCHW

            if is_ann:
                data = data.sum(1) / 15.0  # BCHW
            else:
                data = data.permute(1, 0, 2, 3, 4)  # BTCHW -> TBCHW

            data = data.to(device)
            
            functional.reset_net(encoder)
            feats = encoder(data).cpu()
            data_total.append(feats)
            target_total.append(target)
            i += 1
    return torch.cat(data_total), torch.cat(target_total)


def classification(encoder, train_set, val_set, dataset, is_ann, is_1layer, is_random):
    print("Total features - TRAIN")
    train_data, train_target = Total(encoder, train_set, dataset, is_ann)

    print("Total features - TEST")
    test_data, test_target = Total(encoder, val_set, dataset, is_ann)
    print("Shape of test data:", test_data.shape)
    torch.save(test_data, f"experiments/features/{dataset}_ann{is_ann}_1layer{is_1layer}_random{is_random}.pt")

    print("PCA")
    pca = PCA(n_components=200).fit(train_data, train_target)
    train_data, test_data = pca.transform(train_data), pca.transform(test_data)

    print("SVM")
    target = SVC(C=2.4).fit(train_data, train_target).predict(test_data)
    accuracy = (torch.tensor(target) == test_target).sum() / len(test_target)
    mess = f"Final Accuracy: {accuracy * 100 :.2f}%\n"
    print(mess)
    with open(f"report.txt", "a") as fp:
        fp.write(f"==> DATA={dataset} IS_ANN={is_ann} 1LAYER={is_1layer} RANDOM={is_random}==> ")
        fp.write(mess)
        fp.flush()
    return accuracy
