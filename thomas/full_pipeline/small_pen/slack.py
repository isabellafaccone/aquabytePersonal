import warnings
from io import BytesIO

import matplotlib.pyplot as plt
from keras.callbacks import Callback
from slackclient import SlackClient


class SlackCallback(Callback):
    def __init__(self, token, model_description, channel="U9M2HG5EY", user="#thomas"):
        super(SlackCallback, self).__init__()
        self.channel = channel
        self.model_description = model_description
        self.error = None
        self.ts = None
        self.sc = SlackClient(token)
        self.losses = {}

    def on_train_begin(self, logs=None):
        response = self.send_slack_message(self.channel, self.model_description)
        if response['ok']:
            self.error = False
            self.ts = response['ts']
        else:
            self.error = True
            warnings.warn('Slack error:' + str(response))

    def on_epoch_end(self, epoch, logs=None):
        # send message to the thread
        for (loss, value) in logs.items():
            if loss not in self.losses:
                self.losses[loss] = []
            self.losses[loss].append(value)

        message = 'Epoch {epoch:03d} \n loss:{val_loss:.4f}    val_loss:{val_loss:.4f}'
        # self.send_slack_message(self.channel, message.format(epoch=epoch + 1, **logs))
        out = BytesIO()
        plt.plot(self.losses['loss'], color='r')
        plt.plot(self.losses['val_loss'], color='b')
        plt.legend(['loss', 'val_loss'])
        plt.savefig(fname=out, format='png')
        out.seek(0)
        response = self.send_image(message.format(epoch=epoch + 1, **logs), filename='LearningCurve.png', image=out)

    def send_image(self, message, filename, image):
        response = self.sc.api_call('files.upload', channel=self.channel, as_user=True, filename=filename, file=image)

        # makes the file public
        response = self.sc.api_call("files.sharedPublicURL", file=response['file']['id'])

        attachments = {'attachments': {'fallback': 'Learning curves fallback', 'title': 'Loss plot',
                                       'image_url': response['file']['permalink_public']}}
        response = self.sc.api_call("chat.postMessage", channel=self.channel, text=message, thread_ts=self.ts,
                                     attachments=attachments, as_user=True)

    def send_slack_message(self, channel, text, **kwargs):
        return self.sc.api_call("chat.postMessage", channel=channel, text=text, thread_ts=self.ts, as_user=True, **kwargs)
