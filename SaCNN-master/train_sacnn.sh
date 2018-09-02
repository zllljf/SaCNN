CAFFE=/home/zll/OPT/caffe
TMP=/build/tools/caffe
CAFFE=${CAFFE}${TMP}
SOLVER=/home/zll/OPT/SaCNN-CrowdCounting-Tencent_Youtu/SaCNN-master/solver.prototxt
GPU_LIST=0
LOG=log.txt
## iter ? change ?
WEIGHT=/home/zll/OPT/SaCNN-CrowdCounting-Tencent_Youtu/SaCNN-master/model/Part_B/pretrain_iter_950000.caffemodel
echo "Please input train average loss: "
read average_loss

if [ ! -x "result" ]; then mkdir result; fi
if [ -f $LOG ]; then rm $LOG; fi
$CAFFE train --solver=$SOLVER --gpu=$GPU_LIST 2>&1 | tee $LOG
mv log.txt result/
cp solver.prototxt solver_old.prototxt

if [ ! -x "result_c" ]; then mkdir result_c; fi
if [ -f $LOG ]; then rm $LOG; fi
$CAFFE train --solver=$SNAPSHOT --gpu=$GPU_LIST --weights=$WEIGHT 2>&1 | tee $LOG
mv log.txt result_c/
