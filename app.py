import streamlit.web.cli as stcli
import os
import sys



def streamlit_run():
    print("\n\n\nとらのすけを起こしています...\n\n\n")

    # カレントディレクトリ
    if hasattr(sys, "_MEIPASS"):
        # アプリ実行時はPyInstallerでの一時展開先
        BASE_PATH = sys._MEIPASS
    else:
        # スクリプト実行時はプロジェクトのディレクトリ
        BASE_PATH = os.path.abspath(".")

    src = BASE_PATH + '/tranosuke/main.py'
    sys.argv=['streamlit', 'run', src, '--global.developmentMode=false']
    sys.exit(stcli.main())

if __name__ == "__main__":
    streamlit_run()