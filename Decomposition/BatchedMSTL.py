from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union
import warnings

import numpy as np
import pandas as pd
from scipy.stats import boxcox
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.seasonal import DecomposeResult
from statsmodels.tsa.tsatools import freq_to_period

if TYPE_CHECKING:
    from collections.abc import Sequence
    from statsmodels.tools.typing import ArrayLike1D


class BatchedMSTL:
    """
    Batched and robust MSTL decomposition — supports multiple series at once.
    Safe for short sequences and integrates well with TimeBase orthogonal bases.

    Each series (row of input) is decomposed independently using the same
    seasonal periods and STL parameters.
    """

    def __init__(
        self,
        endog: ArrayLike1D,
        *,
        periods: Optional[Union[int, Sequence[int]]] = None,
        windows: Optional[Union[int, Sequence[int]]] = None,
        lmbda: Optional[Union[float, str]] = None,
        iterate: int = 2,
        stl_kwargs: Optional[dict[str, Union[int, bool, None]]] = None,
    ):
        self.endog = endog
        self._Y = self._to_2d_array(endog)
        self.n_series, self.nobs = self._Y.shape
        self.lmbda = lmbda
        self.periods, self.windows = self._process_periods_and_windows(periods, windows)
        self.iterate = iterate
        self._stl_kwargs = self._remove_overloaded_stl_kwargs(
            stl_kwargs if stl_kwargs else {}
        )

    # ---------------------------------------------------------------------
    def fit(self):
        """
        Decompose each series in the batch into trend, seasonal(s), and residuals.

        Returns
        -------
        list[DecomposeResult]
            One DecomposeResult per input series.
        """
        results = []

        for idx in range(self.n_series):
            y = self._Y[idx]
            nobs = len(y)
            num_seasons = len(self.periods)
            iterate = 1 if num_seasons == 1 else self.iterate

            # --- Box–Cox transform ---
            if self.lmbda == "auto":
                y_trans, lmbda = boxcox(y, lmbda=None)
                self.est_lmbda = lmbda
            elif self.lmbda:
                y_trans = boxcox(y, lmbda=self.lmbda)
            else:
                y_trans = y

            # --- Filter valid periods ---
            valid_periods = [p for p in self.periods if p < nobs / 2]
            if not valid_periods:
                warnings.warn(
                    f"[Series {idx}] No valid periods (< nobs/2={nobs/2}). "
                    "Returning flat components.",
                    UserWarning,
                    stacklevel=2,
                )
                trend = np.full_like(y_trans, np.mean(y_trans))
                resid = y_trans - trend
                seasonal = np.zeros_like(y_trans)
                results.append(
                    DecomposeResult(
                        pd.Series(y_trans, name="observed"),
                        pd.Series(seasonal, name="seasonal"),
                        pd.Series(trend, name="trend"),
                        pd.Series(resid, name="resid"),
                        pd.Series(np.ones_like(y_trans), name="robust_weight"),
                    )
                )
                continue

            # --- Ensure valid windows ---
            windows = tuple(
                min(w, max(7, (nobs // 2) | 1))
                for w in self.windows[: len(valid_periods)]
            )

            # --- Extract STL parameters ---
            stl_inner_iter = self._stl_kwargs.pop("inner_iter", None)
            stl_outer_iter = self._stl_kwargs.pop("outer_iter", None)

            # --- Initialize seasonal matrix ---
            seasonal = np.zeros(shape=(len(valid_periods), nobs))
            deseas = y_trans.copy()

            # --- Iterative decomposition ---
            for _ in range(iterate):
                for i, period in enumerate(valid_periods):
                    try:
                        deseas = deseas + seasonal[i]
                        res = STL(
                            endog=deseas,
                            period=period,
                            seasonal=windows[i],
                            robust=True,
                            **self._stl_kwargs,
                        ).fit(inner_iter=stl_inner_iter, outer_iter=stl_outer_iter)
                        seasonal[i] = res.seasonal
                        deseas = deseas - seasonal[i]
                    except Exception as e:
                        warnings.warn(
                            f"[Series {idx}] ⚠️ Failed STL for period={period}: {e}. Using zeros.",
                            UserWarning,
                            stacklevel=2,
                        )
                        seasonal[i] = np.zeros(nobs)

            # --- Collect results ---
            trend = res.trend if "res" in locals() else np.full_like(y_trans, np.mean(y_trans))
            rw = res.weights if "res" in locals() else np.ones_like(y_trans)
            resid = deseas - trend
            seasonal = np.squeeze(seasonal.T)

            # --- Package into pandas objects ---
            index = None
            if isinstance(self.endog, pd.DataFrame):
                index = self.endog.columns[idx]
            elif isinstance(self.endog, pd.Series):
                index = self.endog.index

            y_s = pd.Series(y_trans, name="observed", index=index)
            trend_s = pd.Series(trend, name="trend", index=index)
            resid_s = pd.Series(resid, name="resid", index=index)
            rw_s = pd.Series(rw, name="robust_weight", index=index)

            cols = [f"seasonal_{p}" for p in valid_periods]
            if seasonal.ndim == 1:
                seasonal_s = pd.Series(seasonal, index=index, name="seasonal")
            else:
                seasonal_s = pd.DataFrame(seasonal, columns=cols, index=index)

            results.append(DecomposeResult(y_s, seasonal_s, trend_s, resid_s, rw_s))

        return results

    # ---------------------------------------------------------------------
    def __str__(self):
        return (
            f"BatchedMSTL(n_series={self.n_series}, "
            f"periods={self.periods}, windows={self.windows}, "
            f"iterate={self.iterate})"
        )

    # ---------------------------------------------------------------------
    def _process_periods_and_windows(
        self,
        periods: Union[int, Sequence[int], None],
        windows: Union[int, Sequence[int], None],
    ) -> tuple[Sequence[int], Sequence[int]]:
        periods = self._process_periods(periods)
        if windows:
            windows = self._process_windows(windows, num_seasons=len(periods))
            periods, windows = self._sort_periods_and_windows(periods, windows)
        else:
            windows = self._process_windows(windows, num_seasons=len(periods))
            periods = sorted(periods)
        if len(periods) != len(windows):
            raise ValueError("Periods and windows must have same length")
        return periods, windows

    # ---------------------------------------------------------------------
    def _process_periods(
        self, periods: Union[int, Sequence[int], None]
    ) -> Sequence[int]:
        if periods is None:
            periods = (self._infer_period(),)
        elif isinstance(periods, int):
            periods = (periods,)
        return periods

    # ---------------------------------------------------------------------
    def _process_windows(
        self,
        windows: Union[int, Sequence[int], None],
        num_seasons: int,
    ) -> Sequence[int]:
        if windows is None:
            windows = self._default_seasonal_windows(num_seasons)
        elif isinstance(windows, int):
            windows = (windows,)
        return windows

    # ---------------------------------------------------------------------
    def _infer_period(self) -> int:
        freq = None
        if isinstance(self.endog, (pd.Series, pd.DataFrame)):
            freq = getattr(self.endog.index, "inferred_freq", None)
        if freq is None:
            raise ValueError("Unable to determine period from endog")
        return freq_to_period(freq)

    # ---------------------------------------------------------------------
    @staticmethod
    def _sort_periods_and_windows(
        periods, windows
    ) -> tuple[Sequence[int], Sequence[int]]:
        if len(periods) != len(windows):
            raise ValueError("Periods and windows must have same length")
        periods, windows = zip(*sorted(zip(periods, windows)))
        return periods, windows

    # ---------------------------------------------------------------------
    @staticmethod
    def _remove_overloaded_stl_kwargs(stl_kwargs: dict) -> dict:
        for arg in ["endog", "period", "seasonal"]:
            stl_kwargs.pop(arg, None)
        return stl_kwargs

    # ---------------------------------------------------------------------
    @staticmethod
    def _default_seasonal_windows(n: int) -> Sequence[int]:
        return tuple(7 + 4 * i for i in range(1, n + 1))

    # ---------------------------------------------------------------------
    @staticmethod
    def _to_2d_array(x: ArrayLike1D) -> np.ndarray:
        """Convert input to (n_series, n_timesteps) 2D array."""
        arr = np.asarray(x, dtype=np.double)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2:
            raise ValueError("Input must be 1D or 2D array")
        return np.ascontiguousarray(arr)
