import os
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int32

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PARQUET_PATH = os.path.join(CURRENT_DIR, "data", "transactions.parquet")

# Entity for user
user = Entity(name="user_id", join_keys=["user_id"], value_type=ValueType.INT64)

# File source for historical data
data_source = FileSource(
    path=PARQUET_PATH,
    event_timestamp_column="event_timestamp",
)

# Feature view
transaction_fv = FeatureView(
    name="transaction_features",
    entities=[user], # entity is the key used to join the features
    ttl=timedelta(days=1), # time to live for the features
    schema=[
        Field(name="amount", dtype=Float32),
        Field(name="transaction_time", dtype=Float32),
        Field(name="user_age", dtype=Int32),
        Field(name="is_international", dtype=Int32),
        Field(name="avg_30d_spending", dtype=Float32),
    ],
    online=True, # online is true when we want to use the features in real-time
    source=data_source, # source is the data source for the features
)