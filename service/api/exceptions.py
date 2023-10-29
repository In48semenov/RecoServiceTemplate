import typing as tp
from http import HTTPStatus


class AppException(Exception):
    def __init__(
        self,
        status_code: int,
        error_key: str,
        error_message: str = "",
        error_loc: tp.Optional[tp.Sequence[str]] = None,
    ) -> None:
        self.error_key = error_key
        self.error_message = error_message
        self.error_loc = error_loc
        self.status_code = status_code
        super().__init__()


# class UserNotFoundError(AppException):
#     def __init__(
#         self,
#         status_code: int = HTTPStatus.NOT_FOUND,
#         error_key: str = "user_not_found",
#         error_message: str = "User is unknown",
#         error_loc: tp.Optional[tp.Sequence[str]] = None,
#     ):
#         super().__init__(status_code, error_key, error_message, error_loc)


# class ModelNotFoundError(AppException):
#     def __init__(
#         self,
#         status_code: int = HTTPStatus.NOT_FOUND,
#         error_key: str = "model_not_found",
#         error_message: str = "Model name is unknown",
#         error_loc: tp.Optional[tp.Sequence[str]] = None,
#     ):
#         super().__init__(status_code, error_key, error_message, error_loc)


class AuthenticateError(AppException):
    def __init__(
        self,
        status_code: int = HTTPStatus.UNAUTHORIZED,
        error_key: str = "token_is_not_correct",
        error_message: str = "Token is not correct",
        error_loc: tp.Optional[tp.Sequence[str]] = None,
    ):
        super().__init__(status_code, error_key, error_message, error_loc)


class ModelNotFoundError(Exception):
    def __init__(
        self, 
        message="Model name is unknown"
    ):
        self.message = message
        super().__init__(self.message)


class UserNotFoundError(Exception):
    def __init__(
        self, 
        message="User is unknown"
    ):
        self.message = message
        super().__init__(self.message)


class DevilsUsersError(Exception):
    def __init__(
        self, 
        message="Userid is devisible by 666"
    ):
        self.message = message
        super().__init__(self.message)
