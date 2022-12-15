import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
from resnet import ResNet18
import time
from load_data import FingerTip


parser = argparse.ArgumentParser(description='PyTorch Robot FingerTip Training')
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints')  #restore result path
parser.add_argument('--net', default='./model/net_010.pth', help="path to net (to continue training)")  #restore training path
# parser.add_argument('--net', default=False, help="path to net (to continue training)")  #restore training path
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dir = "./lazydata/train"
vali_dir = "./lazydata/vali"
test_dir = "./lazydata/test"


EPOCH = 4
LR = 0.001  # learning rate
Milestones = [135, 185]
batch_size = 8
Debug = False

mean = [0.485, 0.456, 0.406, 0.4]
std = [0.229, 0.224, 0.225, 0.2]

train_set = FingerTip(train_dir, mean, std)
vali_set = FingerTip(vali_dir, mean, std)


train_iter = DataLoader(train_set, batch_size=batch_size, shuffle=True)
vali_iter = DataLoader(vali_set, batch_size=batch_size, shuffle=True)

# model-ResNet
net = ResNet18().to(device)


criterion = nn.MSELoss()  # MSE
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9,
                      weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=Milestones, gamma=0.1)


if __name__ == "__main__":
    if args.net:
        net.load_state_dict(torch.load(args.net,map_location=torch.device('cpu')))
        net = net.to(device)

    print("Start Training")
    with open("log.txt", "w") as f2:
        for epoch in range(EPOCH):
            train_loss = 0.0
            val_loss = 0.0

            net.train()
            sum_loss = 0.0
            correct = 0.0
            total = 0
            begin = time.time()

            for i, data in enumerate(train_iter, 0):

                length = len(train_iter)
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                # forward + backward
                outputs = net(inputs)
                loss = criterion(outputs, labels).to(torch.float64)
                loss.backward()
                optimizer.step()
                scheduler.step()

                # print loss for every batch
                sum_loss += loss.item()
                total += labels.size(0)

                print("[Epoch:{}/{}, Batch:{}/{}] Loss: {:.6f}".format(epoch + 1, EPOCH, i + 1,
                                                                                         str(length),
                                                                                         sum_loss / (i + 1)
                                                                                         ))

                f2.write("[Epoch:{}/{}, Batch:{}/{}] Loss: {:.6f}".format(epoch + 1, EPOCH, i + 1,
                                                                                         str(length),
                                                                                         sum_loss / (i + 1)
                                                                                         ))
                f2.write('\n')
                f2.flush()

            train_loss = sum_loss / length

            # test loss for every epoch
            with torch.no_grad():
                sum_loss = 0.0
                correct = 0.0
                total = 0
                for x, y in vali_iter:
                    net.eval()
                    images, labels = x, y
                    images, labels = images.to(device), labels.to(device)
                    outputs = net(images)
                    loss = criterion(outputs, labels)
                    sum_loss += loss.item()
                    total += labels.size(0)

            val_loss = sum_loss / len(vali_iter)
            end = time.time()
            print(
                "[Epoch:{}/{}] Train Loss: {:.6f} Test Loss: {:.6f} | Cost time:{:.2f}min".format(
                    epoch + 1, EPOCH, train_loss, val_loss, (end - begin) / 60.0))
            f2.write(
                "[Epoch:{}/{}] Train Loss: {:.6f} Test Loss: {:.6f} | Cost time:{:.2f}min".format(
                    epoch + 1, EPOCH, train_loss, val_loss, (end - begin) / 60.0))
            f2.write('\n')
            f2.flush()

            # torch.save(net, '%s/net_%03d.pth' % (args.outf, epoch + 1))
            torch.save(net.state_dict(), '%s/net_%03d.pth' % (args.outf, epoch + 1))

            print("Training Finished, Total EPOCH=%d" % EPOCH)
