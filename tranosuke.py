import streamlit.web.cli as stcli
import os
import sys



def streamlit_run():
    # pyinstallerでは絶対パスでの指定が必要
    src = os.path.dirname(sys.executable) + '/main.py'
    sys.argv=['streamlit', 'run', src, '--global.developmentMode=false']
    sys.exit(stcli.main())

if __name__ == "__main__":
    streamlit_run()