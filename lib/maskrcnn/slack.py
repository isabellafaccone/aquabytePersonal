import warnings
from io import BytesIO

import matplotlib.pyplot as plt
from keras.callbacks import Callback
from slackclient import SlackClient


class SlackCallback(Callback):
    def __init__(self, token, channel='#thomas', message='Epoch {epoch:03d} loss:{val_loss:.4f} val_loss:{val_loss:.4f}'
                 , model_description='Start of training', plot=['loss', 'val_loss']):
        super(SlackCallback, self).__init__()
        self.losses = []
        self.error = False

        self.token = token
        self.channel = channel
        
        self.model_description = model_description

        self.ts = None
        self.message = message

        self.best_logs = {}
        self.best_epoch = 1
        self.plot = plot or []

    def on_train_begin(self, logs={}):
        response = self.send_message(self.model_description)
        if response['ok']:
            self.error = False
            self.ts = response['ts']
        else:
            self.error = True
            warnings.warn('Slack error:' + str(response))

    def on_epoch_end(self, epoch, logs={}):
        # send message to the thread
        self.losses.append(logs)
        if self.best_only:
            if self.operation(logs[self.monitor], self.best):
                self.send_message(self.message.format(epoch=epoch + 1, **logs))
                self.best = logs[self.monitor]
                self.best_logs = logs
                self.best_epoch = epoch + 1
        else:
            self.send_message(self.message.format(epoch=epoch, **logs))

    def on_train_end(self, logs=None):
        # Report best results and plot losses
        self.send_message('Best results:\n' + self.message.format(epoch=self.best_epoch, **self.best_logs))
        for p in self.plot:
            plt.plot([log[p] for log in self.losses])

        out = BytesIO()
        plt.savefig(fname=out, format='png')
        out.seek(0)
        response = self.send_image(filename='LearningCurve.png', image=out)
        # plt.show()

    def send_message(self, text, **kwargs):
        try:
            return self.send_slack_message(token=self.token, channel=self.channel, text=text, ts=self.ts,
                                           attachments=kwargs['attachments'])
        except:
            # print('No attachments')
            # print(kwargs)
            pass
        if not self.error:
            return self.send_slack_message(token=self.token, channel=self.channel, text=text, ts=self.ts, **kwargs)

    def send_image(self, filename, image):
        if not self.error:
            response = self.attach_slack_file(token=self.token, channel=self.channel, ts=self.ts, filename=filename,
                                              file=image)
            if response['ok']:
                attachments = {'attachments': {'fallback': 'Learning curves fallback', 'title': 'Learning curves title',
                                               'image_url': response['file']['url_private']}}
                return self.send_slack_attachment(token=self.token, channel=self.channel, text='', ts=self.ts,
                                                  attachments=attachments)

    def send_slack_message(self, token, channel, text, ts=None, **kwargs):
        sc = SlackClient(token)
        return sc.api_call("chat.postMessage", channel=channel, text=text, thread_ts=ts, **kwargs)

    def send_slack_attachment(self, token, channel, text, ts=None, attachments=None):
        sc = SlackClient(token)
        return sc.api_call("chat.postMessage", channel=channel, text=text, thread_ts=ts, attacments=attachments)

    def attach_slack_file(self, token, channel, ts, filename, file, **kwargs):
        sc = SlackClient(token)
        return sc.api_call('files.upload', channel=channel, as_user=True, thread_ts=ts, filename=filename, file=file,
                           **kwargs)