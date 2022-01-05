import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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
    data = pd.read_csv('datasets/avazu_20w.txt')
    
    # Get columns' names
    sparse_features = list(data.columns)
    sparse_features.remove('click')
    sparse_features.remove('id')
    sparse_features.remove('hour')
    target = ['click']

    # Preprocess your data
    data[sparse_features] = data[sparse_features].fillna('-1')
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    

    # Split your dataset
    train, test = train_test_split(data, test_size=0.2)

    train_model_input = {name: train[name] for name in sparse_features}
    test_model_input = {name: test[name] for name in sparse_features}
    

    # Instantiate a FeatureMetas object, 
    # add your features' meta information to it
    feature_metas = FeatureMetas()
    for feat in sparse_features:
        feature_metas.add_sparse_feature(name=feat, one_hot_dim=data[feat].nunique(), embedding_dim=32)

    # ====================== Models that can be selected to run ======================
    # ====== baseline model-> DeepFM, DCN
    # ====== state-of-the-art model-> xDeepFM, FGCNN

    # Instantiate a model and compile it
    # model = DeepFM(
    #     feature_metas=feature_metas,
    #     linear_slots=sparse_features,
    #     fm_slots=sparse_features,
    #     dnn_slots=sparse_features
    # )

    # # Instantiate a model and compile it
    # model = DCN(
    #     feature_metas=feature_metas
    # )

    # Instantiate a model and compile it
    model = xDeepFM(
        feature_metas = feature_metas,
        linear_slots = sparse_features,
        fm_slots = sparse_features,
        dnn_slots = sparse_features
    )

    # Instantiate a model and compile it
    # model = FGCNN(
    #     feature_metas=feature_metas
    # )
    
    # # Instantiate a model and compile it    
    # model =  Combined_Model(
    #     feature_metas = feature_metas, 
    #     linear_slots = sparse_features, 
    #     fm_slots = sparse_features, 
    #     dnn_slots = sparse_features
    #     )


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
