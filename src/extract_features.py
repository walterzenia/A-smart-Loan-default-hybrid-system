import pandas as pd
import logging
from .feature_engineering import (
    process_apps,
    process_prev,
    get_prev_agg,
    process_bureau,
    get_bureau_agg,
    process_pos,
    process_install,
    process_card,
)

logger = logging.getLogger(__name__)

def traditional_features(apps, prev, bureau, bureau_bal):

    apps_all = process_apps(apps)
    prev_agg = get_prev_agg(prev)
    bureau_agg = get_bureau_agg(bureau, bureau_bal)
    logger.info('prev_agg shape: %s bureau_agg shape: %s', prev_agg.shape, bureau_agg.shape)
    logger.info('apps_all before merge shape: %s', apps_all.shape)
    apps_all = apps_all.merge(prev_agg, on='SK_ID_CURR', how='left')
    apps_all = apps_all.merge(bureau_agg, on='SK_ID_CURR', how='left')
    logger.info('apps_all after merge with prev_agg shape: %s', apps_all.shape)


    return apps_all




def hybrid_features(apps,bureau, bureau_bal, prev, pos_bal, install, card_bal):

    apps_all =  process_apps(apps)
    bureau_agg = get_bureau_agg(bureau, bureau_bal)
    prev_agg = get_prev_agg(prev)
    pos_bal_agg = process_pos(pos_bal)
    install_agg = process_install(install)
    card_bal_agg = process_card(card_bal)
    # logger.debug('prev_agg shape: %s bureau_agg shape: %s', prev_agg.shape, bureau_agg.shape)
    logger.info('pos_bal_agg shape: %s install_agg shape: %s card_bal_agg shape: %s', pos_bal_agg.shape, install_agg.shape, card_bal_agg.shape)
    logger.info('apps_all before merge shape: %s', apps_all.shape)

    # Join with apps_all
    apps_all = apps_all.merge(prev_agg, on='SK_ID_CURR', how='left')
    apps_all = apps_all.merge(bureau_agg, on='SK_ID_CURR', how='left')
    apps_all = apps_all.merge(pos_bal_agg, on='SK_ID_CURR', how='left')
    apps_all = apps_all.merge(install_agg, on='SK_ID_CURR', how='left')
    apps_all = apps_all.merge(card_bal_agg, on='SK_ID_CURR', how='left')

    logger.info('apps_all after merge with all shape: %s', apps_all.shape)
    # apps_all = reduce_mem_usage(apps_all) # Apply memory reduction after merging
    print('data types are converted for a reduced memory usage')


    return apps_all

# @title
def behaviorial_features(apps, pos_bal, install, card_bal):

    apps_all =  process_apps(apps)
    pos_bal_agg = process_pos(pos_bal)
    install_agg = process_install(install)
    card_bal_agg = process_card(card_bal)
    # logger.debug('prev_agg shape: %s bureau_agg shape: %s', prev_agg.shape, bureau_agg.shape)
    logger.info('pos_bal_agg shape: %s install_agg shape: %s card_bal_agg shape: %s', pos_bal_agg.shape, install_agg.shape, card_bal_agg.shape)
    logger.info('apps_all before merge shape: %s', apps_all.shape)

    # Join with apps_all
    # apps_all = apps_all.merge(prev_agg, on='SK_ID_CURR', how='left')
    # apps_all = apps_all.merge(bureau_agg, on='SK_ID_CURR', how='left')
    apps_all = apps_all.merge(pos_bal_agg, on='SK_ID_CURR', how='left')
    apps_all = apps_all.merge(install_agg, on='SK_ID_CURR', how='left')
    apps_all = apps_all.merge(card_bal_agg, on='SK_ID_CURR', how='left')

    logger.info('apps_all after merge with all shape: %s', apps_all.shape)

    #apps_all = reduce_mem_usage(apps_all)
    #print('data types are converted for a reduced memory usage')


    return apps_all

def get_apps_all_encoded(apps_all):

    object_columns = apps_all.dtypes[apps_all.dtypes == 'object'].index.tolist()
    for column in object_columns:
        apps_all[column] = pd.factorize(apps_all[column])[0]

    return apps_all
