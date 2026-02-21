select r.r_regionkey from region r semi join nation n on r.r_regionkey = n.n_regionkey;
