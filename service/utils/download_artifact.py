from typing import Dict

import dill
import pandas as pd
import yaml


def get_artifact() -> Dict:
    """
        The function collects launch parameters from config.
    """
    blending = None
    model_artifact_2 = None
    item_idf = None
    bmp = None
    watched = None
    users_inv_mapping = None
    users_mapping = None

    with open('./service/config/inference.cfg.yml') as models_config:
        params = yaml.safe_load(models_config)

    all_models = params["all_models"]
    run_params = params["current_run"]

    type_reco = run_params['type_reco']

    if type_reco == "offline":
        model_artifact_1 = pd.read_csv(
            run_params["artifact"]["offline_reco_path"]
        )
    else:
        with open(run_params["artifact"]["model_path_1"], "rb") as file:
            model_artifact_1 = dill.load(file)

        blending = run_params["artifact"]["blending"]
        if blending:
            with open(run_params["artifact"]["model_path_2"], "rb") as file:
                model_artifact_2 = dill.load(file)

            item_idf = pd.read_csv("./data/kion_train/items_idf.csv")

        bmp = run_params["artifact"]["bmp"]
        interactions = pd.read_csv('./data/kion_train/interactions.csv')
        watched = interactions.groupby('user_id').agg({'item_id': list})
        users_inv_mapping = dict(enumerate(interactions['user_id'].unique()))
        users_mapping = {v: k for k, v in users_inv_mapping.items()}

    cold_user_artifact = pd.read_csv(
        run_params["cold_user"]["offline_reco_path"]
    )["item_id"].tolist()

    return {
        "all_models": all_models,
        "type_reco": type_reco,
        "model_artifact_1": model_artifact_1,
        "blending": blending,
        "model_artifact_2": model_artifact_2,
        "item_idf": item_idf,
        "cold_user_artifact": cold_user_artifact,
        "bmp": bmp,
        "watched": watched,
        "users_inv_mapping": users_inv_mapping,
        "users_mapping": users_mapping,
    }


artifact = get_artifact()
