select n_regionkey, count(distinct n_name) from nation group by n_regionkey;
