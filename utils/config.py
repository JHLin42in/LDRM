#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 14:35:48 2019

@author: aditya
"""

r"""This module provides package-wide configuration management."""
from typing import Any, List

from yacs.config import CfgNode as CN


class CfgTrain(object):
    r"""
    A collection of all the required configuration parameters. This class is a nested dict-like
    structure, with nested keys accessible as attributes. It contains sensible default values for
    all the parameters, which may be overriden by (first) through a YAML file and (second) through
    a list of attributes and values.

    Extended Summary
    ----------------
    This class definition contains default values corresponding to ``joint_training`` phase, as it
    is the final training phase and uses almost all the configuration parameters. Modification of
    any parameter after instantiating this class is not possible, so you must override required
    parameter values in either through ``config_yaml`` file or ``config_override`` list.

    Parameters
    ----------
    config_yaml: str
        Path to a YAML file containing configuration parameters to override.
    config_override: List[Any], optional (default= [])
        A list of sequential attributes and values of parameters to override. This happens after
        overriding from YAML file.

    Examples
    --------
    Let a YAML file named "config.yaml" specify these parameters to override::

        ALPHA: 1000.0
        BETA: 0.5

    >>> _C = Config("config.yaml", ["OPTIM.BATCH_SIZE", 2048, "BETA", 0.7])
    >>> _C.ALPHA  # default: 100.0
    1000.0
    >>> _C.BATCH_SIZE  # default: 256
    2048
    >>> _C.BETA  # default: 0.1
    0.7

    Attributes
    ----------
    """

    def __init__(self, config_yaml: str, config_override: List[Any] = []):

        self._C = CN()

        self._C.MODEL = CN()
        self._C.MODEL.NAME = 'model'
        self._C.MODEL.T_STEP = 5
        self._C.MODEL.SCALE = 4

        self._C.SET = CN()
        self._C.SET.DATASET = 'SR/Syn'
        self._C.SET.ColorSET = 'Color/X1'
        self._C.SET.GPU = [0]
        self._C.SET.DEVICE = 'cuda'
        self._C.SET.TRAIN_PS = [64, 64]
        self._C.SET.VAL_PS = [64, 64]

        self._C.SET.MULTI_GPU = False
        self._C.SET.SAVE_IMG = False
        self._C.SET.RESUME = False

        self._C.SET.EXP_EPOCHS = 0
        self._C.SET.VAL_EPOCHS = 5
        self._C.SET.EPOCHS = [200, 200, 200]
        self._C.SET.BATCHSIZE = [20, 20, 20]
        self._C.SET.LR_INIT = [2e-4, 2e-4, 2e-4]
        self._C.SET.LR_MIN = [1e-6, 1e-6, 1e-6]

        self._C.DIR = CN()
        self._C.DIR.TRAIN = './dataset/train/'
        self._C.DIR.VAL = './dataset/test/'
        self._C.DIR.SAVE = './checkpoints/'

        # Override parameter values from YAML file first, then from override list.
        self._C.merge_from_file(config_yaml)
        self._C.merge_from_list(config_override)

        # Make an instantiated object of this class immutable.
        self._C.freeze()

    def dump(self, file_path: str):
        r"""Save config at the specified file path.

        Parameters
        ----------
        file_path: str
            (YAML) path to save config at.
        """
        self._C.dump(stream=open(file_path, "w"))

    def __getattr__(self, attr: str):
        return self._C.__getattr__(attr)

    def __repr__(self):
        return self._C.__repr__()
