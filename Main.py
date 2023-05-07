import torch
import torch.utils.data as Data
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from Model import NTGEncoder, NTGDecoder, EncoderDecoder
from Train import trainNTG, predictNTG
import Utils


def show_batch(epochs, dataloader):
    for epoch in range(epochs):
        for step, (batch_x, batch_y) in enumerate(dataloader):
            # training
            print("steop:{}, batch_x:{}, batch_y:{}".format(step, batch_x.shape, batch_y.shape))


osm = Utils.get_osm(2)
# osm.draw_graph(figsize=(10.24, 10.24))
# plt.show()

osmg = osm.graph
print(osmg)
# lcs = Utils.get_lcs(osmg)

K = 3
L = 10

encset, decset, validlen = Utils.get_dataset(osmg, K, L)

# print(encset.shape)
# print(decset.shape)
# print(validlen.shape)

torch_dataset = Data.TensorDataset(encset, decset, validlen)

hidden_size, embed_size, num_layers, batch_size = 500, 128, 2, 256
lr, wd, num_epochs, device = 1e-3, 1e-4, 30, torch.device('cuda')

loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=batch_size,
    shuffle=True,
)

# show_batch(1, loader)

pos_vocab = Utils.get_vocab(100)

encoder = NTGEncoder(len(pos_vocab), embed_size, hidden_size, num_layers)
decoder = NTGDecoder(len(pos_vocab), embed_size, hidden_size, num_layers)
net = EncoderDecoder(encoder, decoder)
losses = trainNTG(net, loader, lr, wd, num_epochs, device, pos_vocab)
for i in range(len(losses)):
    losses[i] /= encset.shape[0]

plt.figure(figsize=(7, 5))
plt.plot(losses)
plt.show()

src = encset[0, :, :]
print(src.shape)
output_seq = predictNTG(net, src, pos_vocab, 6, device)
print(output_seq)
