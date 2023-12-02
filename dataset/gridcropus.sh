#preparing for download
cd /data/ziruiw3/ 
mkdir "gridcorpus"
cd "gridcorpus"
mkdir "video" "align"

# download the video
cd "video"
for i in `seq $1 $2`
do
    printf "\n\n------------------------- Downloading $i th speaker video -------------------------\n\n"
    
    #download the video of the ith speaker
    curl "https://spandh.dcs.shef.ac.uk/gridcorpus/s$i/video/s$i.mpg_vcd.zip" > "s$i.zip" 

    if (( $3 == "y" ))
    then
        unzip -q "s$i.zip"
        rm "s$i.zip"

    fi
done
cd ..

# download the align
cd "align"
for i in `seq $1 $2`
do
    printf "\n\n------------------------- Downloading $i th speaker align -------------------------\n\n"
    
    #download the align of the ith speaker
    curl "https://spandh.dcs.shef.ac.uk/gridcorpus/s$i/align/s$i.tar" > "s$i.tar" 

    if (( $3 == "y" ))
    then
        tar -xf "s$i.tar"
        rm "s$i.tar"

    fi
done