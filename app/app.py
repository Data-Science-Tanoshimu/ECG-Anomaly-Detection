from flask import Flask, render_template, Response, url_for
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

##### Instansiate Flask
app = Flask(__name__, static_folder='static', static_url_path='')


##### Calclate anomaly score
# a function to create vectors with a window size using sliding window
def embed(data, window_size):
    vector_list = []
    for i in range(data.size - window_size + 1):
        tmp = data.tolist()[i:i+window_size]
        vector_list.append(tmp)

    return vector_list

# Window size
def w_size():
    return 20

def load_model():
    with open('static/knn_model.pickle', 'rb') as f:
        model = pickle.load(f)

    return model

def load_scaler():
    with open('static/scaler.pickle', 'rb') as f:
        scaler = pickle.load(f)

    return scaler

def normalize_anomaly(anomaly_scores):
    mx = pd.read_csv('static/knn_distance_max.csv', header=None, names=['max'])

    return anomaly_scores.flatten() / mx['max'][0]

# append nan data on the anomaly data to match the data size
def append_nan(data):
    nan_list = [np.nan for _ in range(w_size() - 1)]

    return np.append(nan_list, data)

def calc_anomaly():
    data = pd.read_csv('static/ecg_abnormal.csv', header=None, names=['ecg'])

    scaler = load_scaler()

    data['ecg'] = scaler.transform(data[['ecg']])

    vectors = embed(data['ecg'].values, w_size())

    model = load_model()

    d_data = model.kneighbors(vectors)[0]

    d_data_norm = normalize_anomaly(d_data)

    anomaly_scores_data = append_nan(d_data_norm)

    return anomaly_scores_data

# if anomaly score is 0.18 or larger than 0.18, the data point is anomalous
def calc_anomalous_intervals(data):
    anomaly_points = [1 if i >= 0.18 else 0 for i in data]

    anomaly_start_end = []
    prev_point = 0
    start_p = None
    end_p = None

    for idx, value in enumerate(anomaly_points):
        if value is 1 and prev_point is 0:
            start_p = idx
            prev_point = value

        if value is 0 and prev_point is 1:
            end_p = idx
            prev_point = value

        if start_p is not None and end_p is not None:
            start_end_dict = {'start': start_p, 'end': end_p}
            anomaly_start_end.append(start_end_dict)

            start_p = None
            end_p = None

    return anomaly_start_end



##### Routing
@app.route('/')
def index():
    return render_template('index.html')


@app.route("/graph", methods=['GET'])
def graph():

    return render_template('graph.html')


@app.route("/render_graph")
def render_graph():
    # delete a previous graph
    plt.cla();

    # prepare data
    original = pd.read_csv('static/ecg_abnormal.csv', header=None, names=['ecg'])
    sc = load_scaler()
    original['ecg'] = sc.transform(original[['ecg']])
    anomaly = calc_anomaly()
    anomaly_points = calc_anomalous_intervals(anomaly)

    # plot data
    fig, ax = plt.subplots(2, 1, figsize=(12, 8));
    ax = ax.ravel()

    # original data
    ax[0].plot(original['ecg']);
    ax[0].set_title('Your ECG');
    ax[0].set_ylim(-7, 7);
    ax[0].set_xlim(0, len(original));
    ax[0].set_ylabel('Normalized ECG');
    for idx, anom in enumerate(anomaly_points):
        if idx is 0:
            ax[0].axvspan(anom['start'], anom['end'], color="red", alpha=0.3, label='Anomaly Intervals');
        else:
            ax[0].axvspan(anom['start'], anom['end'], color="red", alpha=0.3);
    ax[0].legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1)

    # anomaly score
    ax[1].plot(anomaly);
    ax[1].set_title('Anomaly Scores');
    ax[1].set_ylim(0, 1);
    ax[1].set_xlim(0, len(original));
    ax[1].set_ylabel('Anomaly (max = 1)');

    # threshold line
    ax[1].hlines(0.18, 0, len(original), "red", linestyles='dashed', alpha=0.5, label='Anomaly Threshold')
    ax[1].legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1)



    plt.tight_layout();
    fig.suptitle('Your ECG and Anomaly Score', fontsize=20)
    plt.subplots_adjust(top=0.9);

    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)

    return Response(output.getvalue(), mimetype='image/png')


@app.route("/realtime", methods=['GET'])
def realtime():

    return render_template('realtime.html')


# This code is a code for running on jupyter notebook
# app.run()

if __name__ == "__main__":
    app.run(debug=True)