import typing as tp

import implicit
import numpy as np
import pandas as pd


def get_offline_reco(
    user_id: int,
    k_recs: int,
    offline_reco: pd.DataFrame,
) -> tp.List[int]:
    """
        The function creates offline recommendation.
    """
    recs = list()
    if user_id in offline_reco['user_id']:
        recs = offline_reco[
            offline_reco['user_id'] == user_id
            ]['item_id'].tolist()

        if len(recs) > k_recs:
            recs = recs[:k_recs]

    return recs


def get_sim_user(
    user: int,
    model: implicit,
    users_mapping: tp.Dict[int, int],
    users_inv_mapping: tp.Dict[int, int],
    k_recs: int,
    bmp: bool,
) -> pd.DataFrame:
    """
        The function find similar users.
    """
    user_id = users_mapping[user]
    recs = model.similar_items(user_id, N=k_recs)

    if bmp:
        recs = list(filter(lambda x: x[0] != users_inv_mapping[user_id], recs))

    else:
        recs = list(filter(lambda x: x[1] < 1, recs))

    return [users_inv_mapping[user] for user, _ in recs]


def get_online_reco(
    user_id: int,
    model: implicit,
    watched: tp.Dict[int, tp.List[int]],
    users_mapping: tp.Dict[int, int],
    users_inv_mapping: tp.Dict[int, int],
    k_recs,
    bmp: bool = False,
    blending: bool = False,
) -> tp.List[int]:
    """
        The function creates online recommendation.
    """
    recs = list()
    if user_id in users_mapping:
        sim_user_id = get_sim_user(
            user_id,
            model,
            users_mapping,
            users_inv_mapping,
            k_recs,
            bmp
        )

        recs = np.array(
            [item for user in sim_user_id for item in watched[user]]
        )
        recs = recs[np.sort(np.unique(recs, return_index=True)[1])].tolist()

        if (len(recs) > k_recs) and (not blending):
            recs = recs[:k_recs]

    return recs


def get_online_blending_reco(
    user_id: int,
    model_tfidf: implicit,
    model_bmp: implicit,
    watched: pd.DataFrame,
    item_idf: np.array,
    users_mapping: tp.Dict[int, int],
    users_inv_mapping: tp.Dict[int, int],
    k_recs
) -> tp.List[int]:
    """
        The function creates online recommendation
        with blending by tfidf algorithm.
    """
    recs = list()
    if user_id in users_mapping:

        recs_tfidf = get_online_reco(
            user_id=user_id,
            model=model_tfidf,
            watched=watched,
            users_mapping=users_mapping,
            users_inv_mapping=users_inv_mapping,
            k_recs=k_recs,
            bmp=False,
            blending=True,
        )

        recs_bmp = get_online_reco(
            user_id=user_id,
            model=model_bmp,
            watched=watched,
            users_mapping=users_mapping,
            users_inv_mapping=users_inv_mapping,
            k_recs=k_recs,
            bmp=True,
            blending=True,
        )

        recs = np.unique(
            np.concatenate(
                (recs_tfidf, recs_bmp)
            )
        )

        mask = np.in1d(item_idf, recs)
        recs = item_idf[mask].tolist()

        if len(recs) > k_recs:
            recs = recs[:k_recs]

    return recs
