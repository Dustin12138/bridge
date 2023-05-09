import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt

class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon # returns mean plus std dev x (random) epsilon
    

class VAE(tf.keras.Model):
    
    def __init__(self, encoder, decoder, last_layer_sampling=False, **kwargs):
        super(VAE, self).__init__(**kwargs)
        # initialise local encoder and decoder
        self.encoder = encoder
        self.decoder = decoder
        self.sample = last_layer_sampling
        
    
    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data) # get mean, variance and sample z from encoder
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.mean_squared_error(data, reconstruction)
            )
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "total_loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }
    
    
    # https: // www.tensorflow.org / api_docs / python / tf / keras / Model  #:~:text=training%20argument%20(boolean)%20in%20call()
    def call(self, x, training=False):
        encoded = self.encoder(x)
        if training:
            latent = encoded[-1]
        else:
            latent = encoded[0]
        decoded = self.decoder(latent)
        if self.sample and training:
            decoded = decoded[-1]
        return decoded
    
class AE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(AE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    
class Classifier(tf.keras.Model):
    def __init__(self, encoder, classifier, **kwargs):
        super(Classifier, self).__init__()
        self.encoder = encoder
        self.classifier = classifier

    def call(self, x):
        encoded = self.encoder(x)
        classification = self.classifier(encoded)
        return classification

    
def vae_loss(data, reconstruction): #y_true, y_pred
    mu, ln_var, z = vae.encoder(data)
    reconstruction_loss = tf.reduce_mean(
        tf.keras.losses.mean_squared_error(data, reconstruction)
    )
    kl_loss = 1 + ln_var - tf.square(mu) - tf.exp(ln_var)
    kl_loss = tf.reduce_mean(kl_loss)
    kl_loss *= -0.5
    total_loss = reconstruction_loss + kl_loss
    return total_loss


def entropy(labels, pred):
    # regularises so the prediction is not just in 1 cluster
    
    py = tf.math.reduce_mean(pred, axis=0)
    
    return - tf.math.reduce_mean(
        py * tf.math.log(py + 1e-10)
    ) 


def conditional_entropy(labels, pred):
    # forces predictions to be confident
    
    return - tf.math.reduce_mean(
        tf.math.reduce_sum(pred * tf.math.log(pred + 1e-10), axis=1)
    ) 


def pseudo_loss(mu, gold_labels=False, lamda=0.5, focal_loss=False):
      
    def weighted_kl_loss(labels, pred):
    
        ent = entropy(labels, pred)
        conditional_ent = conditional_entropy(labels, pred)
        
        return (1 - mu) * conditional_ent - mu * ent
    
    def loss(labels, pred):
        
        if gold_labels:
            # then mu should probably be less than 0.5
            if focal_loss:
                weighted_kl_loss(labels, pred) + lamda * tfa.losses.SigmoidFocalCrossEntropy(
                    reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
                )(labels, pred)
            return weighted_kl_loss(labels, pred) + lamda * tf.keras.losses.CategoricalCrossentropy()(labels, pred)
        
        return weighted_kl_loss(labels, pred)
        
    return loss
    


def sequencify(arr, window):
    '''
    arr: 2D np array where the next row is the data at the next timestep (ie. time dependent data)
    window: sequence length
    '''
    # desired eventual shape
    shape = (arr.shape[0] - window + 1, window, arr.shape[1])
    
    # strides for processor to know how many bytes to skip / stride over to read from memory
    strides = (arr.strides[0],) + arr.strides
    
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)


def get_x(df):
    # only last row with nan
    df.dropna(inplace=True)
    df.drop(columns=" Timestamp", inplace=True)

    
def plot_data(df):
    for col in df.columns:
        plt.title(col)
        plt.hist(df[col], bins=100)
        plt.show()

        
def preprocess(df, unwanted_cols=['Timestamp', 'Normal/Attack'], wanted_cols=None, scaler=None):
    
    df_relevant = get_relevant_df(df, unwanted_cols=unwanted_cols,
                                  wanted_cols=wanted_cols)
    
    if scaler is None:
        scaled_data = scaler.transform(df_relevant).astype(np.float64)
    elif scaler:
        if scaler == True:
            scaler = RobustScaler()
        scaled_data = scaler.fit_transform(df_relevant).astype(np.float64)
    else:
        scaled_data = df_relevant
    
#     del df_relevant
    
    return scaled_data, scaler


def get_relevant_df(df, unwanted_cols, wanted_cols=None):
    
    df = df.rename(columns=lambda x: x.strip())
    
    if not wanted_cols:
        wanted_cols = [col for col in df.columns if col not in unwanted_cols]
    print("Num Relevant Cols:", len(wanted_cols))
    
    return df[wanted_cols]

        
def get_threshold(model, x):
    x_mu, x_log_var, x_hat = model(x)
    
    mu_diff = x_mu - x
    
    return np.max(mu_diff, axis=0), np.min(mu_diff, axis=0), np.max(x_log_var, axis=0), np.min(x_log_var, axis=0)

        
def results(df_test, scaler, model,
            threshold_x_hat_mu, threshold_x_hat_log_var,
            unwanted_cols=[' Timestamp', 'Normal/Attack'], wanted_cols=None):
    
    x_test, _ = preprocess(df_test, unwanted_cols=unwanted_cols, wanted_cols=wanted_cols, scaler=scaler)
    x_test_mu, x_test_log_var, x_test_hat = model(x_test)
    
    y_test = df_test['Normal/Attack'].replace(["A ttack", "Attack", "Normal"], [True, True, False])
    
    t_m_min, t_m_max = threshold_x_hat_mu
    t_v_min, t_v_max = threshold_x_hat_log_var
    
    metric_list, confusion_matrices = evaluate([x_test_mu, x_test_log_var],
                           [threshold_x_hat_mu, threshold_x_hat_log_var],
                           y_test)
    
    results_df = pd.DataFrame(data=metric_list,
                              columns=["acc", "precision", "recall", "f_score", "support"])
    
    print(results_df)
    
    for c, conf_matrix in enumerate(confusion_matrices):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

        plt.xlabel('Predictions', fontsize=18)
        plt.ylabel('Actuals', fontsize=18)
        plt.title('Confusion Matrix {}'.format(c), fontsize=18)
        plt.show()
    
    return metric_list, confusion_matrices

    
def evaluate(data_list, threshold_list, y_true):
    
    anomalies = []
    
    for i, data in enumerate(data_list):
        
        t_min, t_max = threshold_list[i]
        
        truth = tf.cast(data > t_max, tf.int32) + tf.cast(data < t_min, tf.int32)
        breached = (truth != np.zeros(len(t_min)) )
        anomalies.append(breached)
    
    just_one_breached = np.logical_or.reduce(anomalies)
    all_have_to_be_breached = np.logical_and.reduce(anomalies)
    
    anomalies.append(just_one_breached)
    anomalies.append(all_have_to_be_breached)
    
    return get_metrics(anomalies, y_true)


def get_metrics(anomalies, y_true):
    
    metric_list = []
    confusion_matrices = []
    
    for anomaly_list in anomalies:
        
        y_pred = tf.math.reduce_any(anomaly_list, axis=1)
        
        acc = tf.keras.metrics.Accuracy()(y_true, y_pred)
        precision, recall, f_score, support = precision_recall_fscore_support(y_true, y_pred)
        results = acc, precision, recall, f_score, support
        metric_list.append(results)
        
        confusion_m = confusion_matrix(y_true, y_pred)
        confusion_matrices.append(confusion_m)
    
    return metric_list, confusion_matrices


def get_metrics_classic(confusion_matrices):
    results_metrics = []
    for conf_matrix in confusion_matrices:
        tn, fp, fn, tp = conf_matrix.ravel()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        results_metrics.append([precision, recall, f1])

    return pd.DataFrame(results_metrics, columns=["precision", "recall", "f_score"])
