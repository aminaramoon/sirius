select c_nationkey, min(c_custkey), max(c_custkey), sum(c_custkey), count(*)
from customer group by c_nationkey;
