# -*- mode: python ; coding: utf-8 -*-
import torch
import torchvision
import os

torch_path = os.path.dirname(torch.__file__)
torchvision_path = os.path.dirname(torchvision.__file__)

a = Analysis(
    ['photo_organizer_v1.0.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('photo_classifier_model.pth', '.'),
        (torch_path, 'torch'),
        (torchvision_path, 'torchvision')
    ],
    hiddenimports=['torch', 'torchvision'],
    hookspath=[],
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
    name='photo_organizer_v1.0',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
