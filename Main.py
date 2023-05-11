import torch
import torch.utils.data as Data
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from Model import NTGEncoder, NTGDecoder, EncoderDecoder
from Train import trainNTG, predictNTG, train_test
import Utils


def show_batch(epochs, dataloader):
    for epoch in range(epochs):
        for step, (batch_x, batch_y) in enumerate(dataloader):
            # training
            print("steop:{}, batch_x:{}, batch_y:{}".format(step, batch_x.shape, batch_y.shape))


osm1 = Utils.get_osm(1)
osm2 = Utils.get_osm(2)
# osm.draw_graph(figsize=(10.24, 10.24))
# plt.show()

osmg1 = osm1.graph
osmg2 = osm2.graph
osmg = nx.union(osmg1, osmg2)
# lcs = Utils.get_lcs(osmg)
print(osmg)


K = 2
L = 5
hidden_size, embed_size, num_layers, batch_size = 500, 128, 2, 256
lr, wd, num_epochs, device = 1e-3, 1e-4, 20, torch.device('cuda')

encset, decset, validlen = Utils.get_dataset(osmg, K, L)

print(encset.shape)
print(decset.shape)
print(validlen.shape)

dataset_length = encset.shape[0]
train_length = round(dataset_length * 0.8)
test_length = dataset_length - train_length

# torch_dataset = Data.TensorDataset(encset, decset, validlen)
train_idx, test_idx = Data.random_split(encset, [train_length, test_length])

train_encset = encset[train_idx.indices]
train_decset = decset[train_idx.indices]
train_validlen = validlen[train_idx.indices]
train_dataset = Data.TensorDataset(train_encset, train_decset, train_validlen)

test_encset = encset[test_idx.indices]
test_decset = decset[test_idx.indices]
test_validlen = validlen[test_idx.indices]
test_dataset = Data.TensorDataset(test_encset, test_decset, test_validlen)

# train_dataset = torch_dataset[train_idx.indices]
# test_dataset = torch_dataset[test_idx.indices]

train_loader = Data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
)
test_loader = Data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True,
)
# show_batch(1, loader)

pos_vocab = Utils.get_vocab(100)

encoder = NTGEncoder(len(pos_vocab), embed_size, hidden_size, num_layers)
decoder = NTGDecoder(len(pos_vocab), embed_size, hidden_size, num_layers)
net = EncoderDecoder(encoder, decoder)

# test_src = encset[0, :, :]
# test_gt = decset[0, :]
# sum_num = encset.shape[0]
# losses = trainNTG(net, loader, lr, wd, num_epochs, device, pos_vocab, test_src, sum_num, test_gt)
# for i in range(len(losses)):
#     losses[i] /= sum_num

train_losses, test_losses = train_test(net, train_loader, test_loader, lr, wd, num_epochs, device, pos_vocab,
                                       train_length, test_length)

plt.figure(figsize=(7, 5))
plt.plot(train_losses, 'r--', label='train_loss')
plt.plot(test_losses, 'b-.', label='test_loss')
plt.show()

# print(src.shape)
# output_seq = predictNTG(net, src, pos_vocab, 6, device)
# print(output_seq)
