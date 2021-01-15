input="/Users/Ellen/Documents/Github/BGCArgo_BCGyre/TextFiles/goodjma.txt"
while IFS= read -r line
do
echo "$line"
rsync -avzh --delete vdmzrs.ifremer.fr::argo/$line /Users/Ellen/Desktop/ArgoGDAC/dac/jma
done < "$input"
