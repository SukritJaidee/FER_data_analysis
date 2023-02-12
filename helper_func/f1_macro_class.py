## @title Macro F1, Recall Class
class MacroRecall(tf.keras.metrics.Metric):
    def __init__(self, num_classes=2, **kwargs):
        super(MacroRecall,self).__init__(name='macro_recall',**kwargs) # handles base args (e.g., dtype)
        self.num_classes=num_classes
        self.total_cm = self.add_weight("total", shape=(num_classes,num_classes), initializer="zeros")
    def reset_states(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))
    def update_state(self, y_true, y_pred,sample_weight=None):
        self.total_cm.assign_add(self.confusion_matrix(y_true,y_pred))
        return self.total_cm
    def result(self):
        return self.process_confusion_matrix()
    def confusion_matrix(self,y_true, y_pred):
        """
        Make a confusion matrix
        """
        y_true=tf.argmax(y_true,1)
        y_pred=tf.argmax(y_pred,1)
        cm=tf.math.confusion_matrix(y_true,y_pred,dtype=tf.float32,num_classes=self.num_classes)
        return cm
    
    def process_confusion_matrix(self):
        cm=self.total_cm
        diag_part=tf.linalg.diag_part(cm)
        # precision=diag_part/(tf.reduce_sum(cm,0)+tf.constant(1e-15))
        recall=diag_part/(tf.reduce_sum(cm,1)+tf.constant(1e-15))
        # f1=2*precision*recall/(precision+recall+tf.constant(1e-15))
        return tf.reduce_sum(recall)/self.num_classes

class MacroF1(tf.keras.metrics.Metric):
    def __init__(self, num_classes=2, **kwargs):
        super(MacroF1,self).__init__(name='macro_f1',**kwargs) # handles base args (e.g., dtype)
        self.num_classes=num_classes
        self.total_cm = self.add_weight("total", shape=(num_classes,num_classes), initializer="zeros")
    def reset_states(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))
    def update_state(self, y_true, y_pred,sample_weight=None):
        self.total_cm.assign_add(self.confusion_matrix(y_true,y_pred))
        return self.total_cm
    def result(self):
        return self.process_confusion_matrix()
    def confusion_matrix(self,y_true, y_pred):
        """
        Make a confusion matrix
        """
        y_true=tf.argmax(y_true,1)
        y_pred=tf.argmax(y_pred,1)
        cm=tf.math.confusion_matrix(y_true,y_pred,dtype=tf.float32,num_classes=self.num_classes)
        return cm
    def process_confusion_matrix(self):
        cm=self.total_cm
        diag_part=tf.linalg.diag_part(cm)
        precision=diag_part/(tf.reduce_sum(cm,0)+tf.constant(1e-15))
        recall=diag_part/(tf.reduce_sum(cm,1)+tf.constant(1e-15))
        f1=2*precision*recall/(precision+recall+tf.constant(1e-15))
        return tf.reduce_sum(f1)/self.num_classes
