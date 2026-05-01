from fastapi import APIRouter, HTTPException, status

from ..config import settings
from ..rate_limit import limiter
from ..schemas import LoginRequest, TokenResponse
from ..security import create_access_token, verify_password

router = APIRouter(tags=["auth"])


@router.post("/auth/login", response_model=TokenResponse)
@limiter.limit("20/minute")
def login(payload: LoginRequest) -> TokenResponse:
    user = settings.dashboard_users.get(payload.username)
    if not user or not verify_password(payload.password, user.get("password", "")):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    token = create_access_token(
        {
            "sub": payload.username,
            "role": user.get("role", "viewer"),
            "node_id": user.get("node_id"),
        }
    )
    return TokenResponse(
        access_token=token, role=user.get("role", "viewer"), node_id=user.get("node_id")
    )
