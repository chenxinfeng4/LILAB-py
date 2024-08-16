# python -m lilab.multiview_scripts.webgui
from pywebio.input import *
from pywebio.output import *
from pywebio import pin
from pywebio import start_server
from lilab.multiview_scripts.webgui_ballglobal import app as app_ballglobal
from lilab.multiview_scripts.webgui_ballmove import app as app_ballmove
from lilab.multiview_scripts.webgui_rat import app as app_rat
from lilab.multiview_scripts.webgui_tworat import app as app_tworat


def app():
    sc_rat = put_scope("outrat")
    sc_tworat = put_scope("outtworat")
    sc_ballglobal = put_scope("ballglobal")
    sc_ballmove = put_scope("ballmove")
    put_tabs(
        [
            {"title": "Ball global", "content": sc_ballglobal},
            {"title": "Ball move", "content": sc_ballmove},
            {"title": "Rat", "content": sc_rat},
            {"title": "Two Rats", "content": sc_tworat},
        ]
    )
    app_ballglobal("ballglobal")
    app_ballmove("ballmove")
    app_rat("outrat")
    app_tworat("outtworat")


if __name__ == "__main__":
    start_server(app, debug=True, port="44318")
