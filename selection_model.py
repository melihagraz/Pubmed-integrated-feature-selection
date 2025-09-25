import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    RFE,
    SelectFromModel,
    mutual_info_classif,
    mutual_info_regression,
    f_classif,
    f_regression,
)
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import get_scorer
import logging
from pubmed_searcher import SimplePubMedSearcher
import time
import json
import math
import sqlite3
import warnings
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import io
import base64
import requests
import xml.etree.ElementTree as ET


# ---- Agent Code (embedded for Streamlit) ----
"""
Dataclasses for storing information for the trial method run, subsequent result 
as well as overall agent configurations and hyperparameters.

"""
@dataclass
class TrialPlan:
    """Represents a single trial plan for the agent."""

    strategy: str
    params: Dict[str, Any]
    comment: str = ""

@dataclass
class TrialResult:
    """Represents the outcome of a single trial."""
    metric_name: str
    metric_value: float
    metric_std: float
    n_features: int
    selected_features: List[str]
    pipeline_repr: str
    duration_sec: float
    reflection: str = ""

@dataclass
class AgentConfig:
    """Configuration settings for the agent."""
    target_metric: str = "roc_auc"
    target_threshold: Optional[float] = None
    budget_trials: int = 30
    budget_seconds: Optional[int] = None
    cv_splits: int = 5
    random_state: int = 42
    enable_optuna: bool = True
    optuna_timeout_per_trial: Optional[int] = 60
    imbalance_threshold: float = 0.15
    hitl_enabled: bool = False
    hitl_auto_blocklist: List[str] = None

links=["",""]

class ExperimentStore:
    """Manages logging and retrieval of agent trials and artifacts."""

    def __init__(self, db_path: str = "agent_runs.sqlite"):
        self.db_path = db_path
        self._init()

    def _init(self):
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS trials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL,
                plan TEXT,
                result TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS artifacts (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )
        con.commit()
        con.close()

    def log_trial(self, plan: TrialPlan, result: TrialResult):
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        cur.execute(
            "INSERT INTO trials (ts, plan, result) VALUES (?, ?, ?)",
            (time.time(), json.dumps(asdict(plan)), json.dumps(asdict(result))),
        )
        con.commit()
        con.close()

    def save_artifact(self, key: str, value: Dict[str, Any]):
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        cur.execute("REPLACE INTO artifacts (key, value) VALUES (?, ?)", (key, json.dumps(value)))
        con.commit()
        con.close()

    def load_artifact(self, key: str) -> Optional[Dict[str, Any]]:
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        cur.execute("SELECT value FROM artifacts WHERE key=?", (key,))
        row = cur.fetchone()
        con.close()
        return json.loads(row[0]) if row else None

    def dataframe(self) -> pd.DataFrame:
        con = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM trials ORDER BY id ASC", con)
        con.close()
        if not df.empty:
            df["plan"] = df["plan"].apply(json.loads)
            df["result"] = df["result"].apply(json.loads)
        return df

class HumanInTheLoop:
    """Analyzes the input data to inform the agent's strategy."""
    
    def __init__(self, enabled: bool = False, auto_blocklist: Optional[List[str]] = None):
        self.enabled = enabled
        self.auto_blocklist = auto_blocklist or []

    def approve_features(self, selected: List[str]) -> List[str]:
        if not self.enabled:
            return selected
        approved = []
        for f in selected:
            if any(b in f for b in self.auto_blocklist):
                continue
            approved.append(f)
        return approved

class LiteratureEnhancedAgent:
    """An autonomous agent for feature selection with literature analysis."""

    def __init__(
        self,
        config: AgentConfig,
        pubmed_searcher: Optional[SimplePubMedSearcher] = None,
        store: Optional[ExperimentStore] = None,
        disease_context: Optional[str] = None,
        hitl: Optional[HumanInTheLoop] = None,
    ):
        """
        Initializes the agent.
        Args:
            config (AgentConfig): Agent configuration settings.
            pubmed_searcher (SimplePubMedSearcher, optional): PubMed search instance.
            store (ExperimentStore, optional): Database for storing trial history.
            disease_context (str, optional): The disease context for PubMed searches.
            hitl (HumanInTheLoop, optional): Human-in-the-loop component.
        """
        self.cfg = config
        self.disease_context = disease_context
        self.pubmed_searcher = pubmed_searcher
        self.store = store or ExperimentStore()
        self.hitl = hitl or HumanInTheLoop(config.hitl_enabled, config.hitl_auto_blocklist)
        self.best_pipeline: Optional[Pipeline] = None
        self.best_score: float = -np.inf
        self.best_features: List[str] = []
        self.task_is_classification: Optional[bool] = None
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.history: List[Tuple[TrialPlan, TrialResult]] = []
        self.literature_cache: Dict[str, dict] = {}

    def _sense(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Analyzes the input data to inform the agent's strategy."""

        info: Dict[str, Any] = {}
        info["n_samples"] = len(X)
        info["n_features"] = X.shape[1]
        self.task_is_classification = self._is_classification(y)
        info["task"] = "classification" if self.task_is_classification else "regression"
        self.numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        self.categorical_cols = [c for c in X.columns if c not in self.numeric_cols]
        info["n_numeric"] = len(self.numeric_cols)
        info["n_categorical"] = len(self.categorical_cols)
        if self.task_is_classification:
            vc = y.value_counts(normalize=True)
            min_class_ratio = float(vc.min())
            info["min_class_ratio"] = min_class_ratio
            info["imbalanced"] = min_class_ratio < self.cfg.imbalance_threshold
        else:
            info["y_skew"] = float(pd.Series(y).skew())
        try:
            if self.task_is_classification and set(pd.unique(y)) <= {0,1}:
                numeric = X[self.numeric_cols]
                if not numeric.empty:
                    corr = numeric.corrwith(y.astype(float)).abs()
                    info["max_abs_corr"] = float(corr.max())
                    info["leakage_suspect"] = bool(corr.max() > 0.98)
            else:
                numeric = X[self.numeric_cols]
                if not numeric.empty:
                    corr = numeric.corrwith(y).abs()
                    info["max_abs_corr"] = float(corr.max())
                    info["leakage_suspect"] = bool(corr.max() > 0.999)
        except Exception:
            info["max_abs_corr"] = np.nan
            info["leakage_suspect"] = False
        return info

    @staticmethod
    def _is_classification(y: pd.Series) -> bool:
        """Determines if the task is a classification task."""

        unique = pd.unique(y)
        return (pd.api.types.is_integer_dtype(y) and len(unique) <= 20) or pd.api.types.is_object_dtype(y) or pd.api.types.is_bool_dtype(y)

    def _scorer_name(self) -> str:
        """Returns the appropriate scorer name based on the task and config."""

        if self.cfg.target_metric:
            return self.cfg.target_metric
        return "f1_macro" if self.task_is_classification else "r2"

    def _cv(self):
        """Returns the appropriate cross-validation strategy."""

        return StratifiedKFold(self.cfg.cv_splits, shuffle=True, random_state=self.cfg.random_state) if self.task_is_classification else KFold(self.cfg.cv_splits, shuffle=True, random_state=self.cfg.random_state)

    def _plan(self, sense_info: Dict[str, Any], prev_results: List[TrialResult]) -> TrialPlan:
        """Generates the next trial plan based on sensed data and history."""

        n, p = sense_info["n_samples"], sense_info["n_features"]
        imbalanced = sense_info.get("imbalanced", False)
        p_over_n = p / max(1, n)
        if p_over_n > 1.5:
            candidate_family = "l1"
        elif sense_info["n_categorical"] > sense_info["n_numeric"]:
            candidate_family = "mi"
        elif imbalanced and self.task_is_classification:
            candidate_family = "tree"
        else:
            candidate_family = "kbest"

        params = {
            "k": min( max(5, p // 4), max(1, p - 1) ),
            "step": 0.2,
            "C": 1.0,
            "alpha": 0.001,
            "n_estimators": 300,
        }

        comment = f"family={candidate_family}; p/n={p_over_n:.2f}; imbalanced={imbalanced}"
        return TrialPlan(strategy=candidate_family, params=params, comment=comment)

    def _build_preprocessor(self) -> ColumnTransformer:
        """Builds a preprocessor for numeric and categorical features."""

        transformers = []
        if self.numeric_cols:
            transformers.append(("num", Pipeline([("sc", StandardScaler())]), self.numeric_cols))
        if self.categorical_cols:
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.categorical_cols))
        if not transformers:
            return ColumnTransformer([], remainder="passthrough")
        return ColumnTransformer(transformers, remainder="drop")

    def _make_selector(self, plan: TrialPlan, task_is_cls: bool, X: Optional[pd.DataFrame] = None) -> Tuple[str, Any]:
        """Creates a feature selector based on the trial plan."""

        k = plan.params.get("k", 10)
        n_features = X.shape[1] if X is not None else None

        if plan.strategy == "kbest":
            score_fn = f_classif if task_is_cls else f_regression
            links[0] = "https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html"
            links[1] = (
                "https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html"
                if task_is_cls
                else "https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html"
            )
            if n_features is not None and k > n_features:
                k = "all"
            return "sel", SelectKBest(score_func=score_fn, k=k)

        if plan.strategy == "mi":
            score_fn = mutual_info_classif if task_is_cls else mutual_info_regression
            links[0] = "https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html"
            links[1] = (
                "https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html"
                if task_is_cls
                else "https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html"
            )
            if n_features is not None and k > n_features:
                k = "all"
            return "sel", SelectKBest(score_func=score_fn, k=k)

        if plan.strategy == "rfe":
            base = LogisticRegression(max_iter=2000, random_state=self.cfg.random_state) if task_is_cls else LinearRegression()
            links[0] = "https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html"
            links[1] = "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html" if isinstance(base, LogisticRegression) else "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html"
            return "sel", RFE(estimator=base, n_features_to_select=min(k, n_features) if n_features else k, step=plan.params.get("step", 0.2))

        if plan.strategy == "l1":
            links[0] = "https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html"
            if task_is_cls:
                links[1] = "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html"
                model = LogisticRegression(
                    penalty="l1",
                    solver="saga",
                    C=plan.params.get("C", 1.0),
                    max_iter=3000,
                    random_state=self.cfg.random_state
                )
            else:
                links[1] = "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html"
                model = Lasso(
                    alpha=plan.params.get("alpha", 0.001),
                    max_iter=5000,
                    random_state=self.cfg.random_state
                )
            return "sel", SelectFromModel(model, prefit=False)

        if plan.strategy == "tree":
            links[0] = "https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html"
            model = RandomForestClassifier(
                n_estimators=plan.params.get("n_estimators", 300),
                random_state=self.cfg.random_state,
                n_jobs=-1
            ) if task_is_cls else RandomForestRegressor(
                n_estimators=plan.params.get("n_estimators", 300),
                random_state=self.cfg.random_state,
                n_jobs=-1
            )
            links[1] = "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html" if isinstance(model, RandomForestClassifier) else "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html"
            return "sel", SelectFromModel(model, prefit=False)

        if plan.strategy == "variance":
            links[0] = "https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html"
            return "sel", VarianceThreshold(threshold=1e-5)

        return "sel", VarianceThreshold(threshold=0.0)

    def _default_estimator(self) -> BaseEstimator:
        """Provides a default machine learning model."""

        return LogisticRegression(max_iter=2000, random_state=self.cfg.random_state) if self.task_is_classification else LinearRegression()

    def _act_build_pipeline(self, plan: TrialPlan) -> Pipeline:
        """Builds a scikit-learn pipeline from the plan."""

        pre = self._build_preprocessor()
        sel_name, selector = self._make_selector(plan, self.task_is_classification)
        model = self._default_estimator()
        pipe = Pipeline([
            ("prep", pre),
            (sel_name, selector),
            ("model", model),
        ])
        return pipe

    def _evaluate(self, pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> TrialResult:
        """Evaluates a pipeline using cross-validation."""

        metric = self._scorer_name()
        cv = self._cv()
        scores = cross_val_score(pipe, X, y, cv=cv, scoring=metric, n_jobs=-1)
        start_fit = time.time()
        pipe.fit(X, y)
        n_features = self._infer_selected_feature_count(pipe, X)
        selected = self._infer_selected_feature_names(pipe, X)
        duration = (time.time() - start_fit) + 0.0
        return TrialResult(
            metric_name=metric,
            metric_value=float(np.mean(scores)),
            metric_std=float(np.std(scores)),
            n_features=n_features,
            selected_features=selected,
            pipeline_repr=str(pipe),
            duration_sec=duration,
        )

    def _infer_selected_feature_count(self, pipe: Pipeline, X: pd.DataFrame) -> int:
        """Infers the counts of features selected by the pipeline."""

        try:
            sel = pipe.named_steps.get("sel")
            if hasattr(sel, "get_support"):
                Xt = pipe.named_steps["prep"].fit_transform(X)
                return int(sel.fit(Xt, np.zeros(len(X))).get_support().sum())
        except Exception:
            pass
        return X.shape[1]

    def _infer_selected_feature_names(self, pipe: Pipeline, X: pd.DataFrame) -> List[str]:
        """Infers the counts of features selected by the pipeline."""

        try:
            prep: ColumnTransformer = pipe.named_steps["prep"]
            sel = pipe.named_steps.get("sel")
            feature_names = []
            for name, trans, cols in prep.transformers_:
                if name == "remainder":
                    continue
                if hasattr(trans, "get_feature_names_out"):
                    outs = trans.get_feature_names_out(cols)
                else:
                    outs = cols
                feature_names.extend(list(outs))
            if hasattr(sel, "get_support"):
                Xt = prep.transform(X)
                sel.fit(Xt, np.zeros(len(X)))
                mask = sel.get_support()
                if mask is not None and len(feature_names) == len(mask):
                    return [f for f, m in zip(feature_names, mask) if m]
        except Exception:
            pass
        return list(X.columns)

    def _reflect(self, plan: TrialPlan, result: TrialResult, sense_info: Dict[str, Any]) -> TrialPlan:
        """
        Reflects on the previous result to generate a new plan.
        Reflection logic incase PubMed analysis is not enabled.

        """
        if len(self.history) >= 2:
            if result.metric_value < self.best_score + 1e-4:
                if plan.strategy in {"kbest", "mi"}:
                    plan.params["k"] = max(5, int(plan.params.get("k",10) * 1.5))
                    plan.comment += "; reflect: increase k"
                elif plan.strategy in {"l1","tree"}:
                    plan.strategy = "rfe" if plan.strategy == "l1" else "kbest"
                    plan.comment += "; reflect: switch family"
        
        if sense_info.get("imbalanced", False) and self.task_is_classification and self._scorer_name() not in {"roc_auc","average_precision"}:
            self.cfg.target_metric = "roc_auc"
            plan.comment += "; reflect: set metric=roc_auc"
        
        """ PubMed based Literature-informed reflection logic """
        if self.pubmed_searcher and len(result.selected_features) <= 5:  # Limit for rate limiting
            try:
                lit_scores = []
                for feature in result.selected_features[:5]:
                    if feature not in self.literature_cache:
                        lit_result = self.pubmed_searcher.search_simple(feature,disease_context=self.disease_context, max_results=3)
                        self.literature_cache[feature] = lit_result
                    
                    lit_scores.append(self.literature_cache[feature]['evidence_score'])
                
                if lit_scores:
                    avg_lit_score = sum(lit_scores) / len(lit_scores)
                    
                    # Adjust strategy based on literature support
                    if avg_lit_score < 1.5:  # Low literature support
                        if plan.strategy == "kbest":
                            plan.params["k"] = max(5, int(plan.params["k"] * 0.8))
                            plan.comment += f"; lit_adj: reduce k (low_lit={avg_lit_score:.1f})"
                    elif avg_lit_score > 3.0:  # High literature support
                        plan.comment += f"; lit_adj: keep strategy (high_lit={avg_lit_score:.1f})"
                        
            except Exception as e:
                plan.comment += "; lit_adj: error"
        
        return plan

    def _stop_check(self, start_time: float, trials_done: int, last_result: Optional[TrialResult]) -> bool:
        """Checks if a stopping condition has been met."""

        if trials_done >= self.cfg.budget_trials:
            return True
        if self.cfg.budget_seconds is not None and (time.time() - start_time) >= self.cfg.budget_seconds:
            return True
        if self.cfg.target_threshold is not None and last_result is not None:
            if last_result.metric_value >= self.cfg.target_threshold:
                return True
        return False

    def _maybe_optuna_tune(self, plan: TrialPlan, X: pd.DataFrame, y: pd.Series) -> TrialPlan:
        if not self.cfg.enable_optuna:
            return plan
        try:
            import optuna
        except Exception:
            return plan
        # Simplified optuna integration for Streamlit
        return plan

    def run(self, X: pd.DataFrame, y: pd.Series, progress_callback=None) -> Dict[str, Any]:
        """
        Runs the agent's feature selection process.
        Args:
            X (pd.DataFrame): Features.
            y (pd.Series): Target variable.
            progress_callback (Callable, optional): A function to update a progress bar.
        Returns:
            dict: Summary of the agent's run.
        """
        start = time.time()
        sense_info = self._sense(X, y)
        plan = self._plan(sense_info, [])

        trials = 0
        last_result: Optional[TrialResult] = None
        
        while True:
            plan = self._maybe_optuna_tune(plan, X, y)
            pipe = self._act_build_pipeline(plan)
            result = self._evaluate(pipe, X, y)

            approved = self.hitl.approve_features(result.selected_features)
            result.selected_features = approved

            self.store.log_trial(plan, result)
            self.history.append((plan, result))

            if result.metric_value > self.best_score:
                self.best_score = result.metric_value
                self.best_pipeline = pipe
                self.best_features = approved
                self.store.save_artifact(
                    "best",
                    {
                        "metric": result.metric_name,
                        "score": result.metric_value,
                        "n_features": result.n_features,
                        "features": approved,
                        "pipeline": result.pipeline_repr,
                        "plan": asdict(plan),
                    },
                )

            trials += 1
            last_result = result
            
            # Progress callback for Streamlit
            if progress_callback:
                progress_callback(trials, self.cfg.budget_trials, result)

            new_plan = self._reflect(plan, result, sense_info)
            plan = new_plan

            if self._stop_check(start, trials, last_result):
                break

        elapsed = time.time() - start
        return {
            "best_score": self.best_score,
            "best_metric": self._scorer_name(),
            "best_features": self.best_features,
            "trials": trials,
            "elapsed_sec": elapsed,
            "sense_info": sense_info,
            "history_df": self.store.dataframe(),
            "documentation_link": links,
            "literature_cache": self.literature_cache if self.pubmed_searcher else {}
        }