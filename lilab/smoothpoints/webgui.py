# python -m lilab.smoothpoints.webgui
from pywebio.input import * 
from pywebio.output import * 
from pywebio import pin
from pywebio import start_server
from lilab.smoothpoints.webgui_lstm_train import app as app_lstm_train
from lilab.smoothpoints.webgui_lstm_pred import app as app_lstm_pred
from lilab.smoothpoints.webgui_outlierfree_smooth import app as app_outlierfree_smooth
from lilab.smoothpoints.webgui_plotvideo import app as app_plotvideo


def app():
    sc_lstm_train = put_scope('lstm_train')
    sc_lstm_pred = put_scope('lstm_pred')
    sc_outlier_smooth = put_scope('outlier_smooth')
    sc_plot_video = put_scope('plot_video')
    put_tabs([
    {'title': 'LSTM train', 'content': sc_lstm_train},
    {'title': 'LSTM impute', 'content': sc_lstm_pred},
    {'title': "Outlier&Smooth", 'content': sc_outlier_smooth},
    {'title': "Plot video", 'content': sc_plot_video},
    ])
    app_lstm_train('lstm_train')
    app_lstm_pred('lstm_pred')
    app_outlierfree_smooth('outlier_smooth')
    app_plotvideo('plot_video')



if __name__ == '__main__':
    start_server(app, debug=True, port='44319')
