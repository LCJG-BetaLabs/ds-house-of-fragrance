# Databricks notebook source
from pyspark.sql import DataFrame, Column
from pyspark.sql import functions as f
from typing import Optional, List, Callable, Dict, Union
from typing_extensions import Self, TypeAlias
from enum import Enum
from dataclasses import dataclass
import inspect


class FilterType(Enum):
    """REQUIRED_FILTER will get carried over to DEPLOY_FILTER"""

    REQUIRED_FILTER = "REQUIRED_FILTER"
    DEPLOY_FILTER = "DEPLOY_FILTER"


"""Return column of func should evaluate to 0 if the row should be discarded"""
FilterFunc: TypeAlias = Union[
    Callable[[], Column],
    Callable[[Callable], Column],
]


@dataclass
class Filter:
    name: str
    func: FilterFunc
    type: FilterType

    def format_pyspark_column(self) -> str:
        """
        Returns the string representation of
        resultant PySpark column after filtering
        """
        args = inspect.getfullargspec(self.func).args
        if args:
            # declare a dummy udf to pass into self.func to get the pyspark column
            _udf = args[0]
            exec(f"@f.udf" f"\n" f"def {_udf}(*args):" f"\n\t" f"return args")
            return str(self.func(eval(_udf)))
        else:
            return str(self.func())


FILTER_REGISTRY: Dict[str, Filter] = {}
FILTER_FUNC_ARGS: Dict[str, Callable] = {}


def register_filter_udf(name: str):
    def decorator(udf: Callable):
        if name not in FILTER_REGISTRY:
            raise ValueError(f"Filter '{name}' is not registered")
        _filter = FILTER_REGISTRY[name]
        argspec = inspect.getfullargspec(_filter.func)
        if len(argspec.args) != 1:
            signature = inspect.signature(_filter.func)
            raise ValueError(
                f"Expected filter `{name}` to accept one argument, got {len(argspec.args)}: {signature}"
            )
        FILTER_FUNC_ARGS[name] = udf
        print(f"Registered filter udf for '{name}'")
        return udf

    return decorator


def register_filter(name: str, type: FilterType):
    """Your function should return 0 if the row should be discarded"""

    def decorator(func: FilterFunc):
        if type not in FilterType:
            raise ValueError(f"Invalid type {type}, expected one of {list(FilterType)}")
        argspec = inspect.getfullargspec(func)
        if len(argspec.args) > 1:
            signature = inspect.signature(func)
            raise ValueError(
                f"Filter function can only accept 0 or 1 argument, got {len(argspec.args)}: {signature}"
            )
        _filter = Filter(name=name, func=func, type=type)
        FILTER_REGISTRY[name] = _filter
        print(
            f"Registered new filter '{_filter.name}' ({type}): {_filter.format_pyspark_column()}"
        )
        return func

    return decorator


def run_filter(_filter: Filter) -> Column:
    name = _filter.name
    func = _filter.func
    if name in FILTER_FUNC_ARGS:
        arg = FILTER_FUNC_ARGS[name]
        return func(arg)
    return func()


class FilterRunner:
    def __init__(self):
        self.item_master: Optional[DataFrame] = None
        self.filtered_rec: Optional[DataFrame] = None

    def with_item_master(self, item_master: DataFrame) -> Self:
        self.item_master = item_master
        return self

    def run_filter_rules(self, rec: DataFrame):
        """
        Run the defined filter rules and label them in new columns
        ["keep_default", "keep_deploy"].

        Set a filter rule by `add_default_rule()` or `add_deploy_rule()`.
        :param rec:
        :return:
        """
        if not self.item_master:
            raise ValueError(f"item_master must first be set via with_item_master()")

        source_im = (
            self.item_master.toDF(*[f"source_{c}" for c in self.item_master.columns])
            .withColumnRenamed("source_atg_code", "source")
            .repartition(512)
        )
        sim_im = (
            self.item_master.toDF(*[f"sim_{c}" for c in self.item_master.columns])
            .withColumnRenamed("sim_atg_code", "sim")
            .repartition(512)
        )
        rec = (
            rec.repartition(f.col("sim"))
            .join(source_im, how="inner", on="source")
            .join(sim_im, how="inner", on="sim")
        )
        print(f"Total recommendations: {rec.count()}")

        last_column = f.lit(1)
        for filter_type in FilterType:
            type_str = filter_type.value
            rec = rec.withColumn(type_str, last_column)
            for _filter in FILTER_REGISTRY.values():
                if _filter.type != filter_type:
                    continue
                print(f"Running {type_str} filter '{_filter.name}'")
                name = _filter.name
                rec = rec.withColumn(name, run_filter(_filter))
                rec = rec.withColumn(type_str, f.least(f.col(type_str), f.col(name)))
            print(
                f"{type_str} filters remaining: {rec.select(f.sum(type_str)).collect()[0][0]}"
            )
            last_column = f.col(type_str)

        self.filtered_rec = rec

    def get_filtered_df(
        self,
        filter_type: FilterType,
        keep_filter_columns: bool = False,
    ) -> DataFrame:
        if not self.filtered_rec:
            raise ValueError(f"Call run_filter_rules() first.")
        df = self.filtered_rec.filter(f.col(filter_type.value) == 1)
        if not keep_filter_columns:
            filter_cols = [_filter.name for _filter in FILTER_REGISTRY.values()]
            df = df.drop(*filter_cols)
            type_cols = [t.value for t in FilterType]
            df = df.drop(*type_cols)
        return df
