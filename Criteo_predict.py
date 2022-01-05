import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import tensorflow as tf
from core.features import FeatureMetas

from models.DeepFM import DeepFM
from models.DCN import DCN
from models.xDeepFM import xDeepFM
from models.FGCNN import FGCNN
from models.Combined_Model import Combined_Model



if __name__ == "__main__":
    # Setting GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

    # Read dataset
    data = pd.read_csv('datasets/criteo_sample.txt')
    
    # Get columns' names
    features = list(data.columns)
    features.remove('label')
    target = ['label']
    dense_features = features[:13] # (I1,I2,...,I13)
    sparse_features = features[13:] # (C1,C2,...,C26)

    # Preprocess your data
    # sparse features
    data[sparse_features] = data[sparse_features].fillna('-1')
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    # dense features
    # data = data.dropna()
    # for feat in dense_features:
    #     mode_val = data[feat].mode() # mean()? 还是  # data[dense_features] = data[dense_features].fillna(0, )
    #     data[feat] = data[feat].fillna(value = mode_val)
    data[dense_features] = data[dense_features].fillna(0, )
    mms = MinMaxScaler(feature_range=(0, 1)) # scale to (0,1)
    data[dense_features] = mms.fit_transform(data[dense_features])
    
    

    # Split your dataset
    train, test = train_test_split(data, test_size=0.2)
    # mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"])
    # train = mirrored_strategy.experimental_distribute_dataset(train)
    # test = mirrored_strategy.experimental_distribute_dataset(test)

    train_model_input = {name: train[name] for name in features}
    test_model_input = {name: test[name] for name in features}
    # train_model_label= train['label']
    # test_model_label = test['label']
    

    
    # Instantiate a FeatureMetas object, 
    # add your features' meta information to it
    feature_metas = FeatureMetas()
    for feat in sparse_features:
        feature_metas.add_sparse_feature(name=feat, one_hot_dim=data[feat].nunique(), embedding_dim=32)
    for feat in dense_features:
        feature_metas.add_dense_feature(name=feat, dim = 1)
    
    # # ====================== Models that can be selected to run ======================
    # # ====== baseline model-> DeepFM, DCN
    # # ====== state-of-the-art model-> xDeepFM, FGCNN

    # # Instantiate a model and compile it
    # model = DeepFM(
    #     feature_metas=feature_metas,
    #     linear_slots=features,
    #     fm_slots=features,
    #     dnn_slots=features
    # )

    # # # Instantiate a model and compile it
    # # model = DCN(
    # #     feature_metas=feature_metas,
    # # )

    # Instantiate a model and compile it
    # model = xDeepFM(
    #     feature_metas = feature_metas,
    #     linear_slots = features,
    #     fm_slots = features,
    #     dnn_slots = features
    # )

    # Instantiate a model and compile it
    # model = FGCNN(
    #     feature_metas=feature_metas,
    # )
        
        
    # # Instantiate a model and compile it
    

    # with mirrored_strategy.scope():
    model = Combined_Model(
            feature_metas = feature_metas,
            linear_slots = features,
            fm_slots = features,
            dnn_slots = features
        )
    
    model.compile(optimizer="adam",
                      loss="binary_crossentropy",
                      metrics=['binary_crossentropy'])

    # Train the model
    history = model.fit(x=train_model_input,
                        y=train[target].values,
                        batch_size=256,
                        epochs=1,
                        verbose=2,
                        validation_split=0.2)

    # Testing
    pred_ans = model.predict(test_model_input, batch_size=256)
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
