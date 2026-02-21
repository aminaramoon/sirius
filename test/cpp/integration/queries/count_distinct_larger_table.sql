select c_nationkey, count(distinct c_mktsegment) from customer group by c_nationkey;
