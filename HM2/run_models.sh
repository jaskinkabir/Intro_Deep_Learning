source ../env/bin/activate

echo "Alex 10 DP"
echo "-----------------------------------------"
python3.11 model_scripts/alex_10_dp.py

echo "Alex 10 NDP"
echo "-----------------------------------------"
python3.11 model_scripts/alex_10_ndp.py

echo "Alex 100 NDP"
echo "-----------------------------------------"
python3.11 model_scripts/alex_100_ndp.py