from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import pandas as pd


@dataclass(frozen=True)
class BusinessProfile:
    """Structured business risk and trust context for a user-business pair."""

    business_id: Optional[str]
    display_name: str
    verified: bool
    interacted_before: bool
    recent_orders: int
    recent_payments: int
    opted_out: bool
    opted_in: bool
    trust_score: float
    risk_score: float
    reports: int

    def to_dict(self) -> dict[str, object]:
        """Return the profile as a plain dictionary for downstream use."""
        return {
            "business_id": self.business_id,
            "display_name": self.display_name,
            "verified": self.verified,
            "interacted_before": self.interacted_before,
            "recent_orders": self.recent_orders,
            "recent_payments": self.recent_payments,
            "opted_out": self.opted_out,
            "opted_in": self.opted_in,
            "trust_score": self.trust_score,
            "risk_score": self.risk_score,
            "reports": self.reports,
        }


class BusinessService:
    """Assess business trustworthiness using business account and history data."""

    def __init__(self, data_dir: Optional[Union[str, Path]] = None) -> None:
        self._data_dir = self._resolve_data_dir(data_dir)
        self.businesses = self._load_business_accounts()
        self.user_business_history = self._load_user_business_history()

    def _resolve_data_dir(self, data_dir: Optional[Union[str, Path]]) -> Path:
        """Resolve the dataset directory relative to the repository root."""
        if data_dir is None:
            base_dir = Path(__file__).resolve().parents[1]
            return base_dir / "dataset"

        path = Path(data_dir)
        if not path.is_absolute():
            base_dir = Path(__file__).resolve().parents[1]
            path = base_dir / path

        return path.resolve()

    def _load_business_accounts(self) -> pd.DataFrame:
        """Load the business account metadata from the dataset."""
        file_path = self._data_dir / "business_accounts.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Business dataset not found: {file_path}")
        return pd.read_csv(file_path)

    def _load_user_business_history(self) -> pd.DataFrame:
        """Load the user-business interaction history from the dataset."""
        file_path = self._data_dir / "user_business_history.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"User business history dataset not found: {file_path}")
        return pd.read_csv(file_path)

    def _get_business_row(self, business_id: Optional[Union[str, int]]) -> Optional[pd.Series]:
        """Return the business record for the provided business id, if present."""
        if business_id is None or pd.isna(business_id):
            return None

        normalized_id = str(business_id)
        matches = self.businesses[self.businesses["business_id"] == normalized_id]
        if matches.empty:
            return None

        return matches.iloc[0]

    def _get_user_history(self, user_id: Optional[Union[str, int]], business_id: Optional[Union[str, int]]) -> pd.DataFrame:
        """Return the interaction history rows for a user-business pair."""
        if user_id is None or pd.isna(user_id) or business_id is None or pd.isna(business_id):
            return pd.DataFrame(columns=self.user_business_history.columns)

        normalized_user_id = str(user_id)
        normalized_business_id = str(business_id)
        matches = self.user_business_history[
            (self.user_business_history["user_id"] == normalized_user_id)
            & (self.user_business_history["business_id"] == normalized_business_id)
        ]
        return matches.reset_index(drop=True)

    def _describe_orders(self, history: pd.DataFrame) -> int:
        """Estimate recent order-like activity from the user's history."""
        if history.empty:
            return 0

        reasons = history["why_user_knows_account"].fillna("").astype(str).str.lower()
        order_like = reasons.str.contains(
            r"order|delivery|booking|appointment|subscription|purchase|registration|pickup|membership",
            regex=True,
        )
        return int(order_like.sum())

    def _describe_payments(self, history: pd.DataFrame) -> int:
        """Estimate recent payment-related activity from the user's history."""
        if history.empty:
            return 0

        reasons = history["why_user_knows_account"].fillna("").astype(str).str.lower()
        payment_like = reasons.str.contains(
            r"payment|bill|card|wallet|refund|invoice|loan|receipt|maintenance|pay",
            regex=True,
        )
        return int(payment_like.sum())

    def _is_opted_out(self, history: pd.DataFrame) -> bool:
        """Return True when the user has opted out of promotions."""
        if history.empty:
            return False

        allows_promotions = pd.to_numeric(history["allows_promotions"], errors="coerce").fillna(1)
        opted_out_at = history["promotions_opted_out_at"].fillna("").astype(str)
        return bool(((allows_promotions == 0) | (opted_out_at != "")).any())

    def _is_opted_in(self, history: pd.DataFrame) -> bool:
        """Return True when the user is explicitly opted into promotions."""
        if history.empty:
            return False

        allows_promotions = pd.to_numeric(history["allows_promotions"], errors="coerce").fillna(0)
        opted_out_at = history["promotions_opted_out_at"].fillna("").astype(str)
        return bool(((allows_promotions == 1) & (opted_out_at == "")).any())

    def _calculate_trust_score(self, verified: bool, interacted_before: bool, recent_orders: int, recent_payments: int, opted_out: bool, reports: int) -> float:
        """Compute a simple trust score from known signals."""
        score = 50.0
        if verified:
            score += 25
        if interacted_before:
            score += 10
        score += min(10, recent_orders)
        score -= min(8, recent_payments * 2)
        if opted_out:
            score -= 8
        score -= min(20, reports * 2)
        return round(max(0.0, min(100.0, score)), 2)

    def _calculate_risk_score(self, verified: bool, reports: int, recent_payments: int) -> float:
        """Compute a simple risk score from known signals."""
        score = 0.0
        if not verified:
            score += 35
        score += min(25, reports * 5)
        score += min(20, recent_payments * 5)
        return round(max(0.0, min(100.0, score)), 2)

    def get_profile(self, user_id: Optional[Union[str, int]], business_id: Optional[Union[str, int]]) -> BusinessProfile:
        """Return a structured profile describing trust, risk, and interaction signals."""
        business = self._get_business_row(business_id)
        history = self._get_user_history(user_id, business_id)

        if business is None:
            return BusinessProfile(
                business_id=str(business_id) if business_id is not None and not pd.isna(business_id) else None,
                display_name="Unknown",
                verified=False,
                interacted_before=not history.empty,
                recent_orders=0,
                recent_payments=0,
                opted_out=False,
                opted_in=False,
                trust_score=0.0,
                risk_score=100.0,
                reports=0,
            )

        verified = bool(int(business.get("verified", 0)))
        interacted_before = not history.empty
        recent_orders = self._describe_orders(history)
        recent_payments = self._describe_payments(history)
        opted_out = self._is_opted_out(history)
        opted_in = self._is_opted_in(history)
        reports = int(business.get("user_reports_30d", 0))

        return BusinessProfile(
            business_id=str(business["business_id"]),
            display_name=str(business.get("display_name", "Unknown")),
            verified=verified,
            interacted_before=interacted_before,
            recent_orders=recent_orders,
            recent_payments=recent_payments,
            opted_out=opted_out,
            opted_in=opted_in,
            trust_score=self._calculate_trust_score(verified, interacted_before, recent_orders, recent_payments, opted_out, reports),
            risk_score=self._calculate_risk_score(verified, reports, recent_payments),
            reports=reports,
        )

    def is_verified(self, business_id: Optional[Union[str, int]]) -> bool:
        """Backward-compatible helper for the existing pipeline."""
        return self.get_profile(None, business_id).verified

    def get_reports(self, business_id: Optional[Union[str, int]]) -> int:
        """Backward-compatible helper for the existing pipeline."""
        return self.get_profile(None, business_id).reports

    def get_name(self, business_id: Optional[Union[str, int]]) -> str:
        """Backward-compatible helper for the existing pipeline."""
        return self.get_profile(None, business_id).display_name