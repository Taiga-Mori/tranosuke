# tranosuke
とらのすけ

pyinstaller tranosuke.py --onefile --clean --icon=asset/tranosuke_square.ico --collect-all streamlit --collect-all unidic_lite --collect-all pykakasi --add-data "asset:asset" --add-data "unidic-csj-202302:unidic-csj-202302" --add-data "/Users/taigamori/Works/tranosuke/venv/lib/python3.10/site-packages/pykakasi/data:pykakasi/data" --add-data "phoneme_transition_model.onnx:." --additional-hooks-dir=./hooks

pyinstaller tranosuke.spec --clean