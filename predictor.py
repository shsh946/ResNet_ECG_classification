import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
import net


class Predictor():
    def __init__(self, batch_size=16, max_epochs=10, device=None, learning_rate=1e-4):

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model = nn.DataParallel(net.ResNet18().to(self.device))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.loss = torch.nn.CrossEntropyLoss()
        self.epoch = 0

    def fit_dataset(self, data, collate_fn=default_collate):
        while self.epoch < self.max_epochs:
            print('===== training %i =====' % self.epoch)

            # create dataloader for train
            dataloader = DataLoader(dataset=data, batch_size=self.batch_size, collate_fn=collate_fn, shuffle=True)

            # train epoch
            self._run_epoch(dataloader, training=True)
            self.epoch += 1

            # save model after every 4 epochs
            if self.epoch % 4 == 0 and self.epoch != 0:
                self.save("model-" + str(self.epoch))

    def predict_dataset(self, data, collate_fn=None):
        # set model to eval mode
        self.model.eval()

        dataloader = DataLoader(dataset=data, batch_size=self.batch_size, collate_fn=collate_fn)

        ans = []
        pre = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                predict_BG = self.model.forward(x=batch["ecg"].to(self.device))
                predict_BG = predict_BG.squeeze()

                ans_list = batch["label"].tolist()
                ans += ans_list

                # return predict answers
                predict_list = predict_BG.cpu().tolist()
                if type(predict_list[0]) != list:
                    predict_list = [predict_list]
                pre += predict_list

        return pre, ans

    def save(self, path):
        torch.save({
            'epoch': self.epoch + 1,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load(self, path):
        self.epoch = torch.load(path)['epoch']
        self.model.load_state_dict(torch.load(path)['model'])
        self.optimizer.load_state_dict(torch.load(path)['optimizer'])

    def _run_epoch(self, dataloader, training):
        # set model training/evaluation mode with training
        self.model.train(training)

        # run batches for train
        loss = 0

        if training:
            description = 'training'
        else:
            description = "testing"

        # run batches
        trange = tqdm(enumerate(dataloader), total=len(dataloader), desc=description)

        for i, batch in trange:
            predict_BG = self.model.forward(x=batch["ecg"].to(self.device))
            if len(predict_BG.shape) == 1:
                predict_BG = predict_BG.unsqueeze(0)

            batch_loss = self.loss(predict_BG, batch['label'].to(self.device))

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            # accumulate loss
            loss += batch_loss.item()
            trange.set_postfix(loss=loss / (i + 1))


