from pywebio.input import * 
from pywebio.output import * 
from pywebio import pin
from pywebio import start_server
import os.path as osp
from lilab.smoothpoints.biLSTM_point3d_impute_pred import main

scope_msg = 'lstm_pred_messages'

def on_run():
    input_paths = pin.pin['input_textarea_lstm_pred'].split()
    for input_path in input_paths:
        if not osp.exists(input_path):
            with use_scope(scope_msg, clear=True):
                put_error(f'{input_path} does not exist!')
            return
    with use_scope(scope_msg, clear=True):
        put_text(f'Imputing on {len(input_paths)} files...')
        for input_path in input_paths:
            put_text(input_path[-20:])
            msgstring = main(input_path)
            put_text(msgstring)
        put_success('Done!')


def app(parent=None):
    if not parent:
        parent = 'lstm_pred'
        put_scope(parent)

    with use_scope(parent):
        pin.put_textarea('input_textarea_lstm_pred', 
                         label = 'Imputing "points3d.mat" files',
                         rows=6, code={'lineNumbers' : True})
        put_button('Run', onclick=on_run)
        put_scope(scope_msg)
