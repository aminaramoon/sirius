select n_regionkey, count(distinct n_nationkey) from nation group by n_regionkey;
