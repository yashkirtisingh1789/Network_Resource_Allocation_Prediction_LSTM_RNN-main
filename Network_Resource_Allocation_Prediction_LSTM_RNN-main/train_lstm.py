import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
DATASET_DIR=os.path.join(BASE_DIR,"..","dataset")
GRAPHS_DIR=os.path.join(BASE_DIR,"..","graphs")
MODEL_DIR=os.path.join(BASE_DIR,"..","saved_model")

os.makedirs(GRAPHS_DIR,exist_ok=True)
os.makedirs(MODEL_DIR,exist_ok=True)

SEQ_LEN=20
EPOCHS=60
BATCH_SIZE=256
LR=0.001
N_CLASSES=3

FEATURE_COLS=[
"rsrp_dbm","sinr_db","cqi",
"dl_throughput_mbps","ul_throughput_mbps",
"prb_usage","latency_ms","packet_loss_pct",
"hour","power_state"
]

CLASS_NAMES=["Low","Medium","High"]

def load_and_preprocess():
    df=pd.read_csv(os.path.join(DATASET_DIR,"ue_traffic_dataset.csv"))
    X_list=[]
    y_reg=[]
    y_cls=[]

    for ue_id,grp in df.groupby("ue_id"):
        grp=grp.sort_values("timestamp_s")

        X_raw=grp[FEATURE_COLS].values
        y_r=grp["prb_usage"].values
        y_c=grp["resource_label"].values

        scaler=MinMaxScaler()
        X_norm=scaler.fit_transform(X_raw)

        for i in range(SEQ_LEN,len(X_norm)):
            X_list.append(X_norm[i-SEQ_LEN:i])
            y_reg.append(y_r[i])
            y_cls.append(y_c[i])

    X=np.array(X_list)
    y_reg=np.array(y_reg)/66.0
    y_cls=to_categorical(np.array(y_cls),num_classes=N_CLASSES)

    return X,y_reg,y_cls


def build_model():
    inp=Input(shape=(SEQ_LEN,len(FEATURE_COLS)))

    x=Bidirectional(LSTM(64,return_sequences=True))(inp)
    x=BatchNormalization()(x)
    x=Dropout(0.25)(x)

    x=Bidirectional(LSTM(32))(x)
    x=BatchNormalization()(x)
    x=Dropout(0.25)(x)

    shared=Dense(64,activation="relu")(x)

    r=Dense(32,activation="relu")(shared)
    reg_out=Dense(1,activation="sigmoid",name="prb_output")(r)

    c=Dense(32,activation="relu")(shared)
    cls_out=Dense(3,activation="softmax",name="class_output")(c)

    model=Model(inp,[reg_out,cls_out])

    model.compile(
        optimizer=Adam(LR),
        loss={"prb_output":"mse","class_output":"categorical_crossentropy"},
        metrics={"prb_output":["mae"],"class_output":["accuracy"]}
    )

    return model


def main():
    X,yr,yc=load_and_preprocess()

    X_tr,X_te,yr_tr,yr_te,yc_tr,yc_te=train_test_split(
        X,yr,yc,test_size=0.2,random_state=42
    )

    model=build_model()

    callbacks=[
        EarlyStopping(monitor="val_loss",patience=8,restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss",factor=0.5,patience=4)
    ]

    model.fit(
        X_tr,
        {"prb_output":yr_tr,"class_output":yc_tr},
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )

    model_path=os.path.join(MODEL_DIR,"lstm_5g_model.keras")
    model.save(model_path)

    pred_reg,pred_cls=model.predict(X_te,verbose=0)

    y_true=yr_te*66
    y_pred=pred_reg.flatten()*66

    y_true_cls=np.argmax(yc_te,axis=1)
    y_pred_cls=np.argmax(pred_cls,axis=1)

    print("MAE:",mean_absolute_error(y_true,y_pred))
    print("RMSE:",np.sqrt(mean_squared_error(y_true,y_pred)))
    print("R2:",r2_score(y_true,y_pred))

    print(classification_report(y_true_cls,y_pred_cls,target_names=CLASS_NAMES))


if __name__=="__main__":
    main()