# loop over every python file in model_scripts, echo the name of the file and a seperator line, then run the file
source ../env/bin/activate
for file in model_scripts/*.py
do
    echo $file
    echo "-----------------------------------------"
    python3.11 $file
done