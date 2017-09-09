cd ./xgboost;
sh build.sh;
touch ./rabit/lib/flag
[ -e ./rabit/lib/librabit.a ] && cp ./rabit/lib/librabit.a ./rabit/lib/librabit_empty.a && echo -fopenmp > ./rabit/lib/flag;
cd ../;
echo done building library;
