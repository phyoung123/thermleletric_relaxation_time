#! /bin/sh

for x
do
    echo "Converting $x from CRLF to LF"
    tr -d '\015' < "$x" > "tmp.$x"
    mv "tmp.$x" "$x"
done
echo 'Done'
