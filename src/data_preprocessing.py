import pandas as pd
import numpy as np
import os
import logging
from .config import data_path

logger = logging.getLogger(__name__)


def get_balance_data():
    """Read balance-related CSVs from the data directory using config.data_path."""
    pos_path = data_path("POS_CASH_balance")
    ins_path = data_path("installments_payments")
    card_path = data_path("credit_card_balance")

    for p in (pos_path, ins_path, card_path):
        if not os.path.exists(p):
            logger.error("Required data file not found: %s", p)
            raise FileNotFoundError(f"Required data file not found: {p}")

    pos = pd.read_csv(pos_path)
    ins = pd.read_csv(ins_path)
    card = pd.read_csv(card_path)

    return pos, ins, card


def get_dataset():
    """Read main dataset CSVs and return them in a tuple.

    Note: uses keys defined in `src.config.FILES` and concatenates train/test into `apps`.
    """

    # Read training and test files (train/test were previously swapped)
    app_train_path = data_path("application_train")
    app_test_path = data_path("application_test")

    if not os.path.exists(app_train_path) or not os.path.exists(app_test_path):
        logger.error("Application train/test files not found: %s or %s", app_train_path, app_test_path)
        raise FileNotFoundError("Application train/test files not found in data directory")

    app_train = pd.read_csv(app_train_path)
    app_test = pd.read_csv(app_test_path)
    apps = pd.concat([app_train, app_test], ignore_index=True)

    prev_path = data_path("previous_application")
    bureau_path = data_path("bureau")
    bureau_bal_path = data_path("bureau_balance")

    for p in (prev_path, bureau_path, bureau_bal_path):
        if not os.path.exists(p):
            logger.error("Required data file not found: %s", p)
            raise FileNotFoundError(f"Required data file not found: {p}")

    prev = pd.read_csv(prev_path)
    bureau = pd.read_csv(bureau_path)
    bureau_bal = pd.read_csv(bureau_bal_path)

    pos_bal, install, card_bal = get_balance_data()

    return apps, prev, bureau, bureau_bal, pos_bal, install, card_bal
