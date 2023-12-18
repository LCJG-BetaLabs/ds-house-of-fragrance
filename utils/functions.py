import os
from utils.enviroment import BASE_DIR
from databricks.sdk.runtime import spark


def get_season(d):
    d.pop("day", None)
    d.pop("night", None)
    season = get_max_dict_key(d)
    if season in ['fall', 'winter']:
        return "AW"
    else:
        return "SS"


def get_day_night(d):
    day = d["day"]
    night = d["night"]
    if day > night:
        return "day"
    else:
        return "night"


def get_max_dict_key(d):
    d = {key: value if value is not None else 0 for key, value in d.items()}
    return max(d, key=d.get)


def group_accords(accords, grouping):
    for k, v in grouping.items():
        if accords in v:
            return k


def group_notes(note, mapping):
    for k, v in mapping.items():
        if note in v:
            return k


def clean_mapping(mapping, apply_to_all):
    for r in mapping:
        for a in apply_to_all:
            if a in mapping.keys():
                mapping[r] |= mapping[a]
    mapping = {k: v for k, v in mapping.items() if k not in apply_to_all}
    return mapping


def get_group_table(family):
    result_path = os.path.join(BASE_DIR.replace("/dbfs", ""), f"{family}.parquet")
    df = spark.read.parquet(result_path)
    return df
