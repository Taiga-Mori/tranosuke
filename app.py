import sys
from pathlib import Path

import streamlit.web.cli as stcli


def streamlit_run() -> None:
    src = str(Path(__file__).resolve().parent / "tranosuke" / "gui.py")
    sys.argv = ["streamlit", "run", src, "--global.developmentMode=false"]
    sys.exit(stcli.main())


if __name__ == "__main__":
    streamlit_run()
