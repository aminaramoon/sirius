select n_regionkey, avg(n_nationkey), sum(n_nationkey), count(*)
from nation group by n_regionkey;
