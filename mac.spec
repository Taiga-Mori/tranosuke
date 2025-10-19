# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all
import site
import os

block_cipher = None

assert len(site.getsitepackages()) > 0

package_path = site.getsitepackages()[0]
for p in site.getsitepackages():
    if "site-package" in p:
        package_path = p
        break

datas = [
    ('asset', 'asset'),
    ('tranosuke', 'tranosuke'),
    ('/opt/homebrew/bin/ffmpeg', 'ffmpeg'),
    (os.path.join(package_path, "pykakasi/data"), 'pykakasi/data'),
    (os.path.join(package_path, "altair/vegalite/v5/schema/vega-lite-schema.json"), "altair/vegalite/v5/schema/"),
    (os.path.join(package_path, "streamlit/static"), "streamlit/static"),
    (os.path.join(package_path, "streamlit/runtime"), "streamlit/runtime"),
    (os.path.join(package_path, "torchcodec"), "torchcodec"),
    ]

binaries = []

hiddenimports = [
    "pathlib",
    "numpy",
    "pandas",
    "os",
    "pydomino",
    "librosa",
    "soundfile",
    "tranosuke",
    "zipfile",
    "sys",
    "requests",
    "pykakasi",
    "MeCab",
    "re",
    "soundfile",
    "imageio_ffmpeg",
    "faster_whisper",
    "yaml",
    "platform",
    "pyannote",
    "subprocess"
    ]

tmp_ret = collect_all('streamlit')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('unidic_lite')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('pykakasi')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('pyannote')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=['./hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='とらのすけ_Mac',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['./asset/tranosuke.icns'],
)
