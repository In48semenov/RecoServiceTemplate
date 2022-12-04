from typing import Dict, List

import implicit
import pandas as pd


def add_reco_popular(
    k_recs: int,
    curr_recs: List[int],
    popular_items: List[int],
) -> List[int]:
    """
        The function adds popular to the recommendations,
        if this is not enough.
    """
    for item_pop in popular_items:
        if item_pop not in curr_recs:
            curr_recs.append(item_pop)
        if len(curr_recs) == k_recs:
            break
    return curr_recs


def get_offline_reco(
    user_id: int,
    k_recs: int,
    offline_reco: pd.DataFrame,
    popular_items: List[int],
) -> List[int]:
    """
        The function creates offline recommendation.
    """
    if user_id in offline_reco['user_id']:
        recs = offline_reco[
            offline_reco['user_id'] == user_id
            ]['item_id'].tolist()

        if len(recs) > k_recs:
            recs = recs[:k_recs]
        elif len(recs) < k_recs:
            recs = add_reco_popular(k_recs, recs, popular_items)
    else:
        recs = list(popular_items[:k_recs])

    return recs


def recs_mapper(
    user: int,
    model: implicit,
    user_mapping: Dict[int, int],
    user_inv_mapping: Dict[int, int],
    k_recs: int,
    bmp: bool,
) -> pd.DataFrame:
    """
        The function find similar users.
    """
    user_id = user_mapping[user]
    recs = model.similar_items(user_id, N=k_recs)
    result = pd.DataFrame(
        {
            "sim_user_id": [user_inv_mapping[user] for user, _ in recs],
            "sim": [sim for _, sim in recs]
        }
    )

    if bmp:
        return result[result['sim_user_id'] != user]
    else:
        return result[~(result['sim'] >= 1)]


def get_online_reco(
    user_id: int,
    model: implicit,
    watched: pd.DataFrame,
    user_mapping: Dict[int, int],
    user_inv_mapping: Dict[int, int],
    popular_items: List[int],
    k_recs,
    bmp: bool = False,
) -> list:
    """
        The function creates online recommendation.
    """
    if user_id in user_mapping:
        recs = recs_mapper(
            user_id,
            model,
            user_mapping,
            user_inv_mapping,
            k_recs,
            bmp
        ).merge(
            watched, left_on=['sim_user_id'], right_on=['user_id'], how='left'
        ).explode('item_id').sort_values(
            ['sim'], ascending=False
        ).drop_duplicates(
            ['item_id'], keep='first'
        )["item_id"].tolist()

        if len(recs) > k_recs:
            recs = recs[:k_recs]
        elif len(recs) < k_recs:
            recs = add_reco_popular(k_recs, recs, popular_items)
    else:
        recs = list(popular_items[:k_recs])

    return recs


def get_online_blending_reco(
    user_id: int,
    model_tfidf: implicit,
    model_bmp: implicit,
    watched: pd.DataFrame,
    item_idf: pd.DataFrame,
    user_mapping: Dict[int, int],
    user_inv_mapping: Dict[int, int],
    popular_items: List[int],
    k_recs
) -> list:
    """
        The function creates online recommendation
        with blending by tfidf algorithm.
    """
    if user_id in user_mapping:
        recs = pd.concat(
            [
                recs_mapper(
                    user_id,
                    model_tfidf,
                    user_mapping,
                    user_inv_mapping,
                    k_recs,
                    bmp=False
                ).merge(
                    watched,
                    left_on=['sim_user_id'],
                    right_on=['user_id'],
                    how='left'
                ).explode('item_id').sort_values(
                    ['sim'], ascending=False
                ).drop_duplicates(
                    ['item_id'], keep='first'
                )[["item_id"]],

                recs_mapper(
                    user_id,
                    model_bmp,
                    user_mapping,
                    user_inv_mapping,
                    k_recs,
                    bmp=True
                ).merge(
                    watched,
                    left_on=['sim_user_id'],
                    right_on=['user_id'],
                    how='left'
                ).explode('item_id').sort_values(
                    ['sim'], ascending=False
                ).drop_duplicates(
                    ['item_id'], keep='first'
                )[["item_id"]]
            ],
            ignore_index=True
        ).drop_duplicates(
            ["item_id"]
        ).merge(
            item_idf, left_on='item_id', right_on='index', how='left'
        ).sort_values(['idf'], ascending=False)['item_id'].tolist()

        if len(recs) > k_recs:
            recs = recs[:k_recs]
        elif len(recs) < k_recs:
            recs = add_reco_popular(k_recs, recs, popular_items)
    else:
        recs = list(popular_items[:k_recs])

    return recs
