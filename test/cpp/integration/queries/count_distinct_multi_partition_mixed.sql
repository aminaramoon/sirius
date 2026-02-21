select c_nationkey, count(distinct c_mktsegment), min(c_custkey), count(*)
from customer group by c_nationkey;
