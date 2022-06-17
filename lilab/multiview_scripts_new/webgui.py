# python -m lilab.multiview_scripts_new.webgui
from pywebio.input import * 
from pywebio.output import * 
from pywebio import pin
from pywebio import start_server
from lilab.multiview_scripts_new.webgui_ball import app as app_ball
from lilab.multiview_scripts.webgui_rat import app as app_rat
from lilab.multiview_scripts.webgui_tworat import app as app_tworat


def app():
    sc_rat = put_scope('outrat')
    sc_tworat = put_scope('outtworat')
    sc_ball = put_scope('ball')
    put_tabs([
        {'title': 'Ball Predict', 'content': sc_ball},
        {'title': "Rat", 'content': sc_rat},
        {'title': "Two Rats", 'content': sc_tworat},
    ])
    app_ball('ball')
    app_rat('outrat')
    app_tworat('outtworat')


if __name__ == '__main__':
    start_server(app, debug=True, port='44318')
