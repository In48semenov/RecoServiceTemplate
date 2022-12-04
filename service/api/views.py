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
from service.utils.download_artifact import artifact
from service.utils.run_reco import (
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

    if model_name not in artifact["all_models"]:
        raise ModelNotFoundError(
            error_message=f"Model name '{model_name}' not found"
        )

    if user_id > 10 ** 9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")

    k_recs = request.app.state.k_recs

    try:
        if artifact["type_reco"] == "offline":
            recs = get_offline_reco(
                user_id=user_id,
                k_recs=k_recs,
                offline_reco=artifact["model_artifact_1"],
                popular_items=artifact["cold_user_artifact"]
            )
        elif not artifact["blending"]:
            recs = get_online_reco(
                user_id=user_id,
                model=artifact["model_artifact_1"],
                watched=artifact["watched"],
                user_mapping=artifact["users_mapping"],
                user_inv_mapping=artifact["users_inv_mapping"],
                popular_items=artifact["cold_user_artifact"],
                k_recs=k_recs,
                bmp=artifact["bmp"],
            )
        else:
            recs = get_online_blending_reco(
                user_id=user_id,
                model_tfidf=artifact["model_artifact_1"],
                model_bmp=artifact["model_artifact_2"],
                watched=artifact["watched"],
                item_idf=artifact["item_idf"],
                user_mapping=artifact["users_mapping"],
                user_inv_mapping=artifact["users_inv_mapping"],
                popular_items=artifact["cold_user_artifact"],
                k_recs=k_recs,
            )

    except Exception:
        recs = list(artifact["cold_user_artifact"][:k_recs])

    return RecoResponse(user_id=user_id, items=recs)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
