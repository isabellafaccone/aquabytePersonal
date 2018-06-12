#floyd run --gpu --env torch:py2 "apt-get -y install libmysqlclient-dev && find /usr -name mysql.h | cat"
floyd run --gpu --data CexLkNrSyz4DMe3MeaSbeX --env torch:py2 "luarocks install luasql-mysql MYSQL_INCDIR=/usr/include/mysql && th predict_many.lua -net_fname net/net_kitti_fast_-a_train_all.t7 -disp_max 70"
