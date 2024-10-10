from pathlib import Path
from fastapi import Request, HTTPException


AUTHORIZATION_HEADER_KEY = "Authorization"
BEAER_PREFIX = "Bearer "
API_KEYS_FILE_PATH = Path.cwd().joinpath("api-keys.txt")


def validate_api_key(request: Request) -> str:

    # Get the token
    token = request.headers.get(AUTHORIZATION_HEADER_KEY)

    # Return None if the Authorization header is not present
    if token is None:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    # Return None if the value of the Authorization header does not match the Bearer token format
    if not token.startswith(BEAER_PREFIX):
        raise HTTPException(status_code=401, detail="Invalid Authorization header")

    # Get the API key
    api_key = token.removeprefix(BEAER_PREFIX)

    # Validate the API key
    if not is_api_key_valid(api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")

    return api_key


def get_valid_api_keys() -> set[str]:

    # Read lines
    with open(API_KEYS_FILE_PATH, "r") as f:
        lines = f.readlines()

    # Strip the newline character and add each api key to the set
    api_keys = set(line.strip() for line in lines)

    return api_keys


def is_api_key_valid(api_key: str) -> bool:

    # Get the valid api keys
    api_keys = get_valid_api_keys()

    return api_key in api_keys
