select n_regionkey, min(n_nationkey), max(n_nationkey), count(n_nationkey)
from nation group by n_regionkey;
