select min(c_custkey), max(c_custkey) from customer group by c_nationkey, c_mktsegment;
