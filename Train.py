import torch
import torch.nn as nn
import numpy as np


def sequence_mask(X, valid_len, value=0):
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss


def trainNTG(net, data_loader, lr, wd, num_epochs, device, vocab, test_src, sum_num, test_gt):
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    losses = []
    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    loss = MaskedSoftmaxCELoss()
    # loss = nn.MSELoss()
    net.train()

    for epoch in range(num_epochs):
        # net.train()
        # print("epoch:{}".format(epoch))
        loss_epoch = 0
        for index, (X, Y, Y_valid_len) in enumerate(data_loader):
            # print(X[0, :, :], Y[0, :])
            optimizer.zero_grad()
            X, Y, Y_valid_len = X.to(device), Y.to(device), Y_valid_len.to(device)
            # bos = torch.tensor([[0, 0, 1]]).repeat(Y.shape[0], 1, 1).to(device)
            # bos = torch.tensor([[0, 0]]).repeat(Y.shape[0], 1, 1).to(device)
            # dec_input = torch.cat([bos, Y[:, :-1, :]], 1)
            bos = torch.tensor([vocab['[0, 0]']]).repeat(Y.shape[0], 1).to(device)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)
            Y_hat, _ = net(X, dec_input)
            # Y_hat, Y = torch.tanh(Y_hat) * 0.5 + 0.5, torch.tanh(Y) * 0.5 + 0.5
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()
            # l.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1)
            optimizer.step()
            with torch.no_grad():
                loss_epoch += l.sum().item()
        losses.append(loss_epoch)
        print("epoch:{}, avg_loss:{}".format(epoch, loss_epoch / sum_num))
        # if ((epoch + 1) % 5 == 0):
        #     output_seq = predictNTG(net, test_src, vocab, 10, device)
        #     print("predict:{}, label:{}, ground_truth:{}".format(output_seq, test_src, test_gt))

    return losses


def predictNTG(net, src, vocab, num_steps, device):
    net.eval()
    src = src.to(device)
    src = src.unsqueeze(0).permute(1, 0, 2)
    enc_state = net.encoder.init_state(1)
    for i in range(src.shape[0]):
        if (i == 0):
            enc_outputs = net.encoder(src[i], enc_state)
        else:
            enc_outputs = enc_outputs + net.encoder(src[i], enc_state)
    dec_state = net.decoder.init_state(enc_outputs)
    dec_X = torch.unsqueeze(torch.tensor([vocab['[0, 0]']], dtype=torch.long, device=device), dim=0)
    output_seq = []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        if pred == vocab['end']:
            break
        output_seq.append(pred)
    return output_seq


def train_test(net, train_loader, test_loader, lr, wd, num_epochs, device, vocab, train_num, test_num):
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    train_losses = []
    test_losses = []
    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    loss = MaskedSoftmaxCELoss()

    for epoch in range(num_epochs):
        net.train()
        # print("epoch:{}".format(epoch))
        train_loss_epoch = 0
        for index, (X, Y, Y_valid_len) in enumerate(train_loader):
            # print(X[0, :, :], Y[0, :])
            optimizer.zero_grad()
            X, Y, Y_valid_len = X.to(device), Y.to(device), Y_valid_len.to(device)
            bos = torch.tensor([vocab['[0, 0]']]).repeat(Y.shape[0], 1).to(device)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)
            Y_hat, _ = net(X, dec_input)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1)
            optimizer.step()
            with torch.no_grad():
                train_loss_epoch += l.sum().item()
        net.eval()
        test_loss_epoch = 0
        for index, (X, Y, Y_valid_len) in enumerate(test_loader):
            X, Y, Y_valid_len = X.to(device), Y.to(device), Y_valid_len.to(device)
            X_size = X.shape[0]
            enc_state = net.encoder.init_state(batch_size=X_size)
            X = X.permute(1, 0, 2)
            for i in range(X.shape[0]):
                if (i == 0):
                    enc_outputs = net.encoder(X[i], enc_state)
                else:
                    enc_outputs = enc_outputs + net.encoder(X[i], enc_state)
            dec_state = net.decoder.init_state(enc_outputs)
            dec_X = torch.tensor([vocab['[0, 0]']], dtype=torch.long, device=device).repeat(X_size, 1)
            # Y_hat = []
            for i in range(Y.shape[1]):
                output, dec_state = net.decoder(dec_X, dec_state)
                if (i == 0):
                    Y_hat = output
                else:
                    Y_hat = torch.cat((Y_hat, output), dim=1)
                # Y_hat.append(output)
                dec_X = output.argmax(dim=2)

            # Y_hat = np.array(Y_hat)
            # Y_hat = torch.from_numpy(Y_hat)
            # Y_hat = torch.tensor([item.cpu().detach().numpy() for item in Y_hat])

            # Y_hat = Y_hat.to(device).permute(2, 1, 0, 3).squeeze(dim=0)
            # Y_hat = torch.as_tensor(Y_hat).to(device).permute(1, 0, 2)

            l = loss(Y_hat, Y, Y_valid_len)
            test_loss_epoch += l.sum().item()
        train_losses.append(train_loss_epoch / train_num)
        test_losses.append(test_loss_epoch / test_num)
        print("epoch:{}, train_loss:{}, test_loss:{}".format(epoch, train_loss_epoch / train_num,
                                                             test_loss_epoch / test_num))
        # if ((epoch + 1) % 5 == 0):
        #     output_seq = predictNTG(net, test_src, vocab, 10, device)
        #     print("predict:{}, label:{}, ground_truth:{}".format(output_seq, test_src, test_gt))

    return train_losses, test_losses
