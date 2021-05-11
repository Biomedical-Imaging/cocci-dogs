#
#python main.py --sherpa_trial 13 --notsherpa --gpu 0
#
#python main.py --sherpa_trial 10 --notsherpa --gpu 0 --cm
#
#python main.py --sherpa_trial 10 --notsherpa --gpu 0 --cam --model_path Models/10/00001.h5


test () {
    gpu=${1:-1}
    trial=${2:-1}
    /pkg/python/3.6.1-centos7/bin/python3.6 main.py --sherpa_trial $trial --notsherpa --gpu $gpu
    /pkg/python/3.6.1-centos7/bin/python3.6 main.py --sherpa_trial $trial --notsherpa --gpu $gpu --cm
    /pkg/python/3.6.1-centos7/bin/python3.6 main.py --sherpa_trial $trial --notsherpa --gpu $gpu --cam --model_path Models/$trial/00001.h5
}

# cd /Projects/coccidogs/src
#sh run.sh 3 116 # Inception
#sh run.sh 2 92 # Resnet
#sh run.sh 1 101 # Mobilenet only one on arcus 9
#sh run.sh 1 109 # Shallow
#sh run.sh 0 127 # VGG
test $1 $2