from resnet import se_resnet_18
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import datetime
from tensorboardX import SummaryWriter
import time
import sys, time


class Trainer(object):
    def __init__(self, mode, optim, scheduler, model, config, model_dir):
        assert mode in ['training', 'inference']
        self.mode = mode
        self.model = model
        self.cuda = torch.cuda.is_available()
        self.model_dir = model_dir
        self.optim = optim
        self.epoch = 0
        self.config = config
        self.scheduler = scheduler
        self.set_log_dir()

    def train(self, train_loader, val_loader, loss_function, num_epochs):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dataloaders = {'train': train_loader, 'val': val_loader}

        writer = SummaryWriter(log_dir=self.log_dir)#tensorboard 日志保存地址
        self.model = self.model.to(device)
        for epoch in range(self.epoch, num_epochs):
            since = time.time()
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            for phase in ['train', 'val']:
                if phase == 'train':
                    if self.scheduler:
                        self.scheduler.step()
                    self.model.train()
                else:
                    self.model.eval()
                running_loss = 0.0
                bar_steps = len(dataloaders[phase])
                process_bar = ShowProcess(bar_steps)#定义一个进度条
                total = 0

                aTP=0
                aFP=0
                aTN=0
                aFN=0
                #########################################################
                #               
                #########################################################
                for i, data in enumerate(dataloaders[phase], 0):
                    inputs, labels = data['image'], data['gt_map']
                    inputs, labels = inputs.to(device), labels.to(device)
                    self.optim.zero_grad()
                    #forward
                    #track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        preds = torch.round(outputs)
                        loss = loss_function(outputs, labels)
                        #backward+optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optim.step()

                    # statistics
                    total += inputs.size(0)
                    running_loss += loss.item() * inputs.size(0)
		    #评价指标计算Precession,Recall,F1-Score,MR,Acc
                    FP=preds-labels
                    FP=torch.where(FP==1,FP,torch.zeros(preds.size()).to(device))
                    TP=preds-FP
                    FN=labels-TP
                    TN=1-FN-FP-TP
                    aTP=torch.sum(TP)+aTP
                    aFP=torch.sum(FP)+aFP
                    aTN=torch.sum(TN)+aTN
                    aFN=torch.sum(FN)+aFN
                    process_bar.show_process()

                process_bar.close()
                epoch_loss = running_loss / total
                
                P=aTP/(aTP+aFP)
                R=aTP/(aTP+aFN)
                F1=2*P*R/(P+R)
                Mr=(aFP+aFN)/(aTP+aTN+aFN+aFP)
                Acc=(aTP+aTN)/(aTP+aTN+aFN+aFP)
                
                print('{} Loss: {:.4f} P:{:.4} R:{:.4} F1:{:.4} Mr:{:.4} Acc:{:.4}'.format(phase, epoch_loss, P, R, F1, Mr,Acc))
                writer.add_scalar('{}_loss'.format(phase), epoch_loss, epoch)#保存loss到Tensorboard
                writer.add_scalar('{}_P'.format(phase), P, epoch)#保存Precession
                writer.add_scalar('{}_R'.format(phase), R, epoch)#保存Recall
                writer.add_scalar('{}_F1'.format(phase), F1, epoch)#保存F1
                writer.add_scalar('{}_Mr'.format(phase), Mr, epoch)#保存MR,误判率

            time_elapsed = time.time() - since#训练一个epoch计算消耗时间
            print('one epoch complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            ##############################################################
            #            save the model for every epoch                  #
            ##############################################################
            torch.save({
                'epoch': epoch,'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optim.state_dict(),
                'lr_scheduler': self.scheduler.state_dict(),'loss': loss
            }, self.checkpoint_path.format(epoch))
        writer.close()
        print("train finished")

    def set_log_dir(self, model_path=None):
        """Set the model log directory and epoch counter.
        model_path:If None ,or a format different form what this code uses then set a new 
        log directory and start epochs from 0. Otherwise,extract  the log directory and 
        the epoch counter form the file name.
        """
        if self.mode == 'training':
            now = datetime.datetime.now()
            #if we hanbe a model path with date and epochs use them
            if model_path:
                # Continue form we left of .Get epoch and date form the file name
                # A sample model path might look like:
                #/path/to/logs/coco2017.../DeFCN_0001.h5
                import re
                regex = r".*/[\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/model\_[\w-]+(\d{4})\.pt"
                m = re.match(regex, model_path)
                if m:
                    now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),int(m.group(4)), int(m.group(5)))
                    # Epoch number in file is 1-based, and in Keras code it's 0-based.
                    # So, adjust for that then increment by one to start from the next epoch
                    #self.epoch = int(m.group(6))  + 1
                    print('Re-starting from epoch %d' % self.epoch)

                    # Directory for training logs
            self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(self.config.NAME.lower(), now))
            # Create log_dir if not exists
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)

            # Path to save after each epoch. Include placeholders that get filled by Keras.
            self.checkpoint_path = os.path.join(self.log_dir,"model_{}_*epoch*.pt".format(self.config.NAME.lower()))
            self.checkpoint_path = self.checkpoint_path.replace("*epoch*", "{:04d}")

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
               model directory.
        Returns:
            the path of the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)

        if not dir_names:
            import errno
            raise FileNotFoundError(errno.ENOENT, "Could not find model directory under {}".format(self.model_dir))
        # Pick last directory
        if self.mode == 'training':
            dir_name = os.path.join(self.model_dir, dir_names[-2])
            print(dir_name)
            os.rmdir(os.path.join(self.model_dir, dir_names[-1]))
            
        else:
            dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("model"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            import errno
            raise FileNotFoundError(errno.ENOENT,"Could not find weight files in {}".format(dir_name))
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return checkpoint

    def load_weights(self, file_path, by_name=False, exclude=None):
        """load the weights from the file_path in CNN model.
        """
        checkpoint = torch.load(file_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        #when loading a model on a GPU that was trained and saved on GPU,you
        #should convert the initialized model to a CUDA optimized model using
        #model.to(torch.device("cuda"))
        if self.cuda:
            self.device = torch.device("cuda")
            self.model.to(self.device)

        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.epoch = checkpoint['epoch'] + 1
        self.loss = checkpoint['loss']
        self.set_log_dir(file_path)
        print("load weights from {} finished.".format(file_path))

    def detect(self, image):
        """Runs the detection pipeline.
                images: List of images, potentially of different sizes.
                Returns  a mask of image.
        """
        import numpy as np
        assert self.mode == "inference", "Create model in inference mode."
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        with torch.no_grad():
            image=image.transpose((2, 0, 1))

            inputs = torch.Tensor(image)
            inputs = torch.unsqueeze(inputs,0)
            inputs = inputs.to(device)

            outputs = self.model(inputs)
            preds = torch.round(outputs)
            preds = preds.to('cpu')
            preds = torch.squeeze(preds, 0)  
            #preds = preds.transpose(2,0)
            preds = torch.squeeze(preds,0)
            preds=preds.numpy()
            return preds


    def evaluate(self, val_laoder):
        classes=['Compact_high-rise',
                 'Compact_miderise',
                 'Compact_low-rise',
                 'Open_high-rise',
                 'Open_midrise',
                 'Open_low-rise',
                 'Lightweight_low-rise',
                 'Large_low-rise',
                 'Sparsely_built',
                 'Heavy_industry',
                 'Dense_trees',
                 'Scattred_trees',
                 'Bush_and_scrub',
                 'Low_plants',
                 'Bare_rock_or_paved',
                 'Bare_soil_or_sand',
                 'Water']
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        class_correct = list(0. for i in range(17))
        class_total = list(0. for i in range(17))

        with torch.no_grad():
            for data in val_laoder:
                images,labels=data['sen'],data['label']
                images,labels=images.to(device),labels.to(device)

                outputs = self.model(images)
                _,predicted=torch.max(outputs,1)
                c=(predicted==labels).squeeze()
                for i in range (images.size(0)):
                    label = labels[i]
                    class_correct[label]+=c[i].item()
                    class_total[label]+=1
        for i in range(17):
            print("Accuracy of %5s :%2d %%" % (
                classes[i],100*class_correct[i]/class_total[i]))
class ShowProcess():
    """
    显示处理进度的类
    调用该类相关函数即可实现处理进度的显示
    """
    i = 0  # 当前的处理进度
    max_steps = 0  # 总共需要处理的次数
    max_arrow = 50  #进度条的长度

    # 初始化函数，需要知道总共的处理次数
    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.i = 0
        # 显示函数，根据当前的处理进度i显示进度
        # 效果为[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>]100.00%
    def show_process(self, i=None):
        if i is not None:
            self.i = i
        else:
            self.i += 1
        num_arrow = int(self.i * self.max_arrow / self.max_steps)  #计算显示多少个'>'
        num_line = self.max_arrow - num_arrow  #计算显示多少个'-'
        percent = self.i * 100.0 / self.max_steps  #计算完成进度，格式为xx.xx%
        process_bar = '[' + '>' * num_arrow + '-' * num_line + ']'\
            + '%.2f' % percent + '%' + '\r' #带输出的字符串，'\r'表示不换行回到最左边
        sys.stdout.write(process_bar)  #这两句打印字符到终端
        sys.stdout.flush()

    def close(self, words='done'):
        print('')
        #print(words)
        self.i = 0


if __name__ == "__main__":
    """Here is an example to show how to impliement the class trainer."""
