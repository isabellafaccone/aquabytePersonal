from keras.callbacks import Callback
import json
import math

class SaveHistory(Callback):
    """
    Saves the metric at each end of epoch in 
    json file
    
    Input:
        - json_path
    """
    
    def __init__(self, json_path):
        self.json_path = json_path
    
    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        with open(self.json_path, 'w') as f:
            json.dump(self.history, f)
            
def step_decay(epoch):
    """
    Learning rate scheduler
    '"""
    initial_lrate = 1e-5
    drop = 0.5
    epochs_drop = 20.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

class MAP_eval(Callback):
    def __init__(self, validation_data):
        self.validation_data = validation_data
        self.maps = []

    def eval_map(self):
        x_val, y_true = self.validation_data
        y_pred = self.model.predict(x_val)
        y_pred = list(np.squeeze(y_pred))
        zipped = zip(y_true, y_pred)
        zipped.sort(key=lambda x:x[1],reverse=True)

        y_true, y_pred = zip(*zipped)
        k_list = [i for i in range(len(y_true)) if int(y_true[i])==1]
        score = 0.
        r = np.sum(y_true).astype(np.int64)
        for k in k_list:
            Yk = np.sum(y_true[:k+1])
            score += Yk/(k+1)
        score/=r
        return score

    def on_epoch_end(self, epoch, logs={}):
        score = self.eval_map()
        print "MAP for epoch %d is %f"%(epoch, score)
        self.maps.append(score)
