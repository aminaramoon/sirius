select n_regionkey, max(cast(n_nationkey as Decimal(18,2))) as max_d
from nation group by n_regionkey;