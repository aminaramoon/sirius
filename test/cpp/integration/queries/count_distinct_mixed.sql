select n_regionkey, count(distinct n_nationkey), min(n_nationkey), count(*)
from nation group by n_regionkey;
