max=212
for i in `seq 2 $max`
do
    sed -i -e '$a\ ' README.md
    git remote add origin https://github.com/kaist-dmlab/MG-TAR.git
    git add *
    git commit -m "Update README.md"
    git push origin
    git config credential.helper store
    git config credential.helper cache
    git config credential.helper 'cache --timeout=180000'
done



