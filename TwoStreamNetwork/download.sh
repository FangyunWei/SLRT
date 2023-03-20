#Download Phoenix-2014 and Phoenix-2014T
wget https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/phoenix-2014.v3.tar.gz -O data/phoenix-2014/phoenix-2014.v3.tar.gz
tar -xvf data/phoenix-2014/phoenix-2014.v3.tar.gz data/phoenix-2014/

wget https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/phoenix-2014-T.v3.tar.gz -O data/phoenix-2014t/phoenix-2014-T.v3.tar.gz
tar -xvf data/phoenix-2014t/phoenix-2014-T.v3.tar.gz data/phoenix-2014t/
#Please download CSL-Daily by yourself from http://home.ustc.edu.cn/~zhouh156/dataset/csl-daily/ and place it under data/csl-daily

#Preprocess video files (for dataloader)
zip -r -j data/phoenix-2014/phoenix-2014-videos.zip data/phoenix-2014/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-210x260px/
zip -r -j data/phoenix-2014t/phoenix-2014t-videos.zip data/phoenix-2014t/phoenix-2014t.v3/PHOENIX-2014-T/features/fullFrame-210x260px/
zip -r -j data/csl-daily/csl-daily-videos.zip data/csl-daily/sentence_frames-512x512/

#Download pretrained word embeddings
gdown #?


