from typing import List

import yaml
from fastapi import APIRouter, Depends, FastAPI, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from service.api.exceptions import (
    AuthenticateError,
    ModelNotFoundError,
    UserNotFoundError,
)
from service.config.responses_cfg import example_responses
from service.log import app_logger
from service.utils.download_artifact import artifact_run, popular_item
from service.utils.run_reco import (
    add_reco_popular,
    get_offline_reco,
    get_online_blending_reco,
    get_online_reco,
)

with open('./service/envs/authentication_env.yaml') as env_config:
    ENV_TOKEN = yaml.safe_load(env_config)


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


router = APIRouter()
auth_scheme = HTTPBearer(auto_error=False)


async def authorization_by_token(
    token: HTTPAuthorizationCredentials = Security(auth_scheme),
):
    if token is not None and token.credentials == ENV_TOKEN['token']:
        return token.credentials
    else:
        raise AuthenticateError()


@router.get(
    path="/health",
    tags=["Health"],
)
async def health(
    token: HTTPAuthorizationCredentials = Depends(authorization_by_token)
) -> str:
    return "I am alive"


@router.get(
    path="/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=RecoResponse,
    responses=example_responses,
)
async def get_reco(
    request: Request,
    model_name: str,
    user_id: int,
    token: HTTPAuthorizationCredentials = Depends(authorization_by_token),
) -> RecoResponse:
    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")

    if model_name not in artifact_run["registered_model"]:
        raise ModelNotFoundError(
            error_message=f"Model name '{model_name}' not found"
        )

    if user_id > 10 ** 9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")

    k_recs = request.app.state.k_recs

    try:
        if artifact_run["type_reco"] == "offline":
            recs = get_offline_reco(
                user_id=user_id,
                k_recs=k_recs,
                offline_reco=artifact_run["offline_reco"],
            )
        else:
            if not artifact_run["blending"]:
                recs = get_online_reco(
                    user_id=user_id,
                    model=artifact_run["model"],
                    watched=artifact_run["watched"],
                    users_mapping=artifact_run["users_mapping"],
                    users_inv_mapping=artifact_run["users_inv_mapping"],
                    k_recs=k_recs,
                    bmp=artifact_run["bmp"]
                )

            else:
                recs = get_online_blending_reco(
                    user_id=user_id,
                    model_tfidf=artifact_run["model_tfidf"],
                    model_bmp=artifact_run["model_bmp"],
                    watched=artifact_run["watched"],
                    item_idf=artifact_run["item_idf"],
                    users_mapping=artifact_run["users_mapping"],
                    users_inv_mapping=artifact_run["users_inv_mapping"],
                    k_recs=k_recs,
                )

        if len(recs) != k_recs:
            recs = add_reco_popular(
                k_recs=k_recs,
                curr_recs=recs,
                popular_items=popular_item
            )

    except Exception:
        recs = popular_item[:k_recs]

    return RecoResponse(user_id=user_id, items=recs)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
