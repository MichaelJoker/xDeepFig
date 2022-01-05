import tensorflow as tf
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from core.features import FeatureMetas, Features
from core.blocks import DNN, CIN, InnerProduct, FGCNNlayer
from core.utils import split_tensor, group_embedded_by_dim, group_embedded_by_dim_simulate


def Combined_Model(
        feature_metas,
        linear_slots,
        fm_slots,
        dnn_slots,
        fg_filters=(14, 16, 18, 20),
        fg_widths=(7, 7, 7, 7),
        fg_pool_widths=(2, 2, 2, 2),
        fg_new_feat_filters=(3, 3, 3, 3),
        embedding_initializer='glorot_uniform',
        embedding_regularizer=tf.keras.regularizers.l2(1e-5),
        fixed_embedding_dim=8,
        fm_fixed_embedding_dim=None,
        fm_kernel_initializer='glorot_uniform',
        fm_kernel_regularizer=None,
        linear_use_bias=True,
        linear_kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-4, seed=1024),
        linear_kernel_regularizer=tf.keras.regularizers.l2(1e-5),
        dnn_hidden_units=(128, 64, 1),
        dnn_activations=('relu', 'relu', None),
        dnn_use_bias=True,
        dnn_use_bn=False,
        dnn_dropout=0,
        dnn_kernel_initializers='glorot_uniform',
        dnn_bias_initializers='zeros',
        dnn_kernel_regularizers=tf.keras.regularizers.l2(1e-5),
        dnn_bias_regularizers=None,
        name='xDeepFM_fgcnn'
    ):
    
    # model begin
    with tf.name_scope(name):

        features = Features(metas=feature_metas) # define the class


        
        # Linear Part： Use raw features directly and use wx+b to generate the output
        with tf.name_scope('Linear'):
            linear_output = features.get_linear_logit(use_bias=linear_use_bias,
                                                      kernel_initializer=linear_kernel_initializer,
                                                      kernel_regularizer=linear_kernel_regularizer,
                                                      embedding_group='dot_embedding',
                                                      slots_filter=linear_slots)

        #  CIN Part: Implement high order explicit feature interactions and make prediction
        with tf.name_scope('FM'):
            
            #print(features.values())
            raw_feats = features.get_stacked_feature(
                embedding_group='raw',
                fixed_embedding_dim=fixed_embedding_dim,
                embedding_initializer=embedding_initializer,
                embedding_regularizer=embedding_regularizer,
                slots_filter=fm_slots
            )
            #print(raw_feats)
            
            ## Use FGCNN for inputs of CIN
            # input for feature generation initial
            fg_inputs = features.get_stacked_feature(
                embedding_group='fgcnn',
                fixed_embedding_dim=fixed_embedding_dim,
                embedding_initializer=embedding_initializer,
                embedding_regularizer=embedding_regularizer,
                slots_filter=fm_slots
            )

            fg_inputs = tf.expand_dims(fg_inputs, axis=-1)  # change to (21, 8, 1) like a picture
            # print(fg_inputs)

            new_feats_list = list()
            name = 0
            # fg_new_feat_filters: for recombiner (Fully connected layer)
            for filters, width, pool, new_filters in zip(fg_filters, fg_widths, fg_pool_widths, fg_new_feat_filters):
                # i = i+1
                fg_inputs, new_feats = FGCNNlayer(
                    filters=filters,
                    kernel_width=width,
                    pool_width=pool,
                    new_feat_filters=new_filters
                )(fg_inputs)
                # print(new_feats.shape[1])
                # for i in fg_inputs.shape[1]:
                #     feature_metas.add_sparse_feature(name=name, one_hot_dim=data[feat].nunique(), embedding_dim=32)
                new_feats_list.append(new_feats)
            
            # concat the new generated feat and raw_feat
            inputs = tf.concat(new_feats_list + [raw_feats], axis=1)  # (None, 75, 8)           
            
            fm_dim_groups = split_tensor(inputs, axis=1)  # list: 75
            fm_dim_groups = group_embedded_by_dim_simulate(fm_dim_groups) # Simulate Inputs for CIN
           
            
            fms = [CIN(
                kernel_initializer=fm_kernel_initializer,
                kernel_regularizer=fm_kernel_regularizer
            )(group) for group in fm_dim_groups.values() if len(group) > 1]

            fm_output = tf.add_n(fms)
        
        
        # DNN Part: Use MLP for implicit feature interaction and make prediction
        with tf.name_scope('DNN'):
            # raw inputs initializaer
            raw_feats = features.get_stacked_feature(
                embedding_group='raw',
                fixed_embedding_dim=fixed_embedding_dim,
                embedding_initializer=embedding_initializer,
                embedding_regularizer=embedding_regularizer,
                slots_filter=dnn_slots
            )

            # input for feature generation initial
            fg_inputs = features.get_stacked_feature(
                embedding_group='fgcnn',
                fixed_embedding_dim=fixed_embedding_dim,
                embedding_initializer=embedding_initializer,
                embedding_regularizer=embedding_regularizer,
                slots_filter=dnn_slots
            )

            fg_inputs = tf.expand_dims(fg_inputs, axis=-1)  # change to (21, 8, 1) like a picture
            # print(fg_inputs)

            new_feats_list = list()
            name = 0
            # fg_new_feat_filters: for recombiner (Fully connected layer)
            for filters, width, pool, new_filters in zip(fg_filters, fg_widths, fg_pool_widths, fg_new_feat_filters):
                # i = i+1
                fg_inputs, new_feats = FGCNNlayer(
                    filters=filters,
                    kernel_width=width,
                    pool_width=pool,
                    new_feat_filters=new_filters
                )(fg_inputs)
                # print(new_feats.shape[1])
                # for i in fg_inputs.shape[1]:
                #     feature_metas.add_sparse_feature(name=name, one_hot_dim=data[feat].nunique(), embedding_dim=32)
                new_feats_list.append(new_feats)

            # concat the new generated feat and raw_feat
            inputs = tf.concat(new_feats_list + [raw_feats], axis=1)  # (None, 75, 8)
            inputs = split_tensor(inputs, axis=1)  # list: 75
            # inner product
            inputs_fm = InnerProduct(require_logit=False)(inputs)
            dnn_inputs = tf.concat(inputs + [inputs_fm], axis=1)  # the final new features

            dnn_output = DNN(
                    units=dnn_hidden_units,
                    use_bias=dnn_use_bias,
                    activations=dnn_activations,
                    use_bn=dnn_use_bn,
                    dropout=dnn_dropout,
                    kernel_initializers=dnn_kernel_initializers,
                    bias_initializers=dnn_bias_initializers,
                    kernel_regularizers=dnn_kernel_regularizers,
                    bias_regularizers=dnn_bias_regularizers
                )(dnn_inputs)
        
        output = tf.add_n([linear_output, fm_output, dnn_output])
        output = tf.keras.activations.sigmoid(output)
        # print(output)
        model = tf.keras.Model(inputs=features.get_inputs_list(), outputs=output)


    return model