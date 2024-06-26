import keras
import torch.nn as nn
import torch.nn.functional as F
import xgboost
import openml
import numpy as np
from openxai.model import LoadModel
from openxai.dataloader import ReturnLoaders
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer


def largest_power_of_2_in_sqrt(n):
    return int(2**np.floor(np.log2(np.sqrt(n))))


def get_data_and_model_openxai(data_name):
    _, loader_test = ReturnLoaders(data_name=data_name, download=False, batch_size=128)
    x_test = loader_test.dataset.data
    y_test = loader_test.dataset.targets.to_numpy()
    model = LoadModel(data_name=data_name, ml_model="ann", pretrained=True)
    model.eval()
    return x_test, y_test, model


def get_data_and_model_openml(task_id, random_state=0, fit=True):
    task = openml.tasks.get_task(task_id, download_data=True, download_splits=True, download_features_meta_data=True, download_qualities=False)
    dataset = task.get_dataset()
    data = dataset.get_data()
    df = data[0]
    categorical_features = np.array(data[2])[df.columns != dataset.default_target_attribute]
    # drop duplicate rows
    df = df.drop_duplicates()
    x, y = df.drop(dataset.default_target_attribute, axis=1).to_numpy(), df[dataset.default_target_attribute].to_numpy()
    ## remove columns with a single value
    id_keep =  ~np.all(x[1:] == x[:-1], axis=0)
    x = x[:, id_keep]
    categorical_features = categorical_features[id_keep]
    ## remove columns with all distinct values
    try:
        id_keep = np.apply_along_axis(func1d=lambda x: len(np.unique(x)), axis=0, arr=x) > 1
    except: # feature has nan causing error
        id_keep = np.apply_along_axis(func1d=lambda x: len(np.unique(x.astype(str))), axis=0, arr=x) > 1
    x = x[:, id_keep]
    categorical_features = categorical_features[id_keep]
    ## fix categorical features
    if task_id in [14952, 146195]:
        x = x.astype(int)
    elif task_id in [361261, 361252, 361269, 361267, 361234, 361257, 7592, 14965, 361272, 361268]:
        target_encode_features = np.where(categorical_features)[0]
    ## split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=random_state)
    ## fix categorical features
    if task_id in [361261, 361252, 361269, 361267, 361234, 361257, 361272, 361268]:
        target_encoder = make_column_transformer((
            TargetEncoder(target_type="continuous", random_state=random_state), target_encode_features), 
            remainder="passthrough"
        )
        y_train = y_train.astype(np.float64)
        y_test = y_test.astype(np.float64)
        x_train = target_encoder.fit_transform(x_train, y_train).astype(np.float32)
        x_test = target_encoder.transform(x_test).astype(np.float32)
    elif task_id in [7592, 14965]:
        target_encoder = make_column_transformer((
            TargetEncoder(target_type="binary", random_state=random_state), target_encode_features), 
            remainder="passthrough"
        )
        x_train = target_encoder.fit_transform(x_train, y_train).astype(np.float32)
        x_test = target_encoder.transform(x_test).astype(np.float32)
    ## impute missing data with average
    imputer = SimpleImputer()
    imputer.fit(x_train)
    x_train = imputer.transform(x_train)
    x_test = imputer.transform(x_test)
    ## standardize data
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train).astype(float)
    x_test = scaler.transform(x_test).astype(float)
    ## transform target
    y_train, y_test = _transform_y(y_train, y_test, task_type=task.task_type)
    ## train a model
    n_features = x_train.shape[1]
    if fit:
        if n_features < 32:
            model = _fit_xgboost(x_train, y_train, task_type=task.task_type, random_state=random_state)
            print(model.score(x_test, y_test))
        else:
            model_raw = _fit_neural_network(x_train, y_train, task_type=task.task_type, random_state=random_state)
            model = _convert_keras_to_torch(model_raw, x_train)
        return x_test, y_test, model
    else:
        return x_test, y_test, None


def _transform_y(y_train, y_test, task_type):
    if task_type == 'Supervised Classification':
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)
    elif task_type == 'Supervised Regression':
        standard_scaler = StandardScaler()
        y_train = standard_scaler.fit_transform(y_train.reshape(-1, 1))
        y_test = standard_scaler.transform(y_test.reshape(-1, 1))
    return y_train, y_test


def _fit_xgboost(x_train, y_train, task_type, random_state=0):
    if task_type == 'Supervised Classification':
        model = xgboost.XGBClassifier(n_estimators=200, random_state=random_state)
        model.fit(x_train, y_train)
    elif task_type == 'Supervised Regression':
        model = xgboost.XGBRegressor(n_estimators=200, random_state=random_state)
        model.fit(x_train, y_train)
    return model


def _fit_neural_network(x_train, y_train, task_type, random_state=0):
    keras.utils.set_random_seed(random_state)
    optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True, use_ema=True, ema_momentum=0.99, weight_decay=0.01)
    model = keras.Sequential()
    model.add(keras.layers.Dense(units=128, activation='relu', name="fc1"))
    model.add(keras.layers.Dense(units=64, activation='relu', name="fc2"))
    if task_type == 'Supervised Classification':
        label_encoder = OneHotEncoder(sparse_output=False)
        y_train = label_encoder.fit_transform(y_train.reshape(-1, 1))
        model.add(keras.layers.Dense(units=y_train.shape[1], activation='softmax', name="fc3"))
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    elif task_type == 'Supervised Regression':
        model.add(keras.layers.Dense(units=1, activation='linear', name="fc3"))
        model.compile(loss='mean_squared_error', optimizer=optimizer)
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=4, min_delta=0.005)
    model.fit(
        x_train, 
        y_train, 
        validation_split=0.1, 
        epochs=20, 
        batch_size=int(np.min([x_train.shape[0]/20, 256])), 
        callbacks=[callback], 
        verbose=False
    )
    return model


def _convert_keras_to_torch(model, x):
    new_model = ArtificialNeuralNetwork(x.shape[1], model.output_shape[1])
    layers = [v.value for v in model.variables]
    state = {
        'fc1.weight': layers[0].T,
        'fc1.bias': layers[1],
        'fc2.weight': layers[2].T,
        'fc2.bias': layers[3],
        'fc3.weight': layers[4].T,
        'fc3.bias': layers[5],
    }
    new_model.load_state_dict(state)
    return new_model


class ArtificialNeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        if self.fc3.out_features > 1:
            self.predict_proba = lambda x: self.forward(x)
        else:
            self.predict = lambda x: self.forward(x)

    def forward(self, x):
        x = self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))
        if self.fc3.out_features > 1:
            x = F.softmax(x, dim=1) 
        return x