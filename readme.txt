------------Before train--------------------

A. replace specified file noted in replace/readme.txt
B. compile caffe:
	make clean
	make all
	make pycaffe
C. add python interface because there are .py writed by ourselves.
If you have any questions, please move to:https://blog.csdn.net/zllljf/article/details/81670143

D. do datasets
	there is a datasets sample in ~/SaCNN-master/ShanghaiTech/Part_B

-----------Train-----------------------------
A. cd ~/SaCNN-master
   sudo sh train_sacnn.sh

B. the average_loss is equal to your train datasets, more explanation can be seen in 	Caffe website 


---------------Retrain----------------------
A. after train, there are log.txt in ~/result and .caffemodel in ~/result_c
   You can draw loss-iter curve by using tools provided by Caffe which in ~/result

B. copy the .caffemodel to ~/model/mine
C. cd ~caffe
   ./build/tools/caffe train --solver ~/solverretrain.prototxt --gpu 0 --weights /~/model/mine/~.caffemodel


-----------------Test----------------------
A. After retrain, there is a .caffemodel in /result
B. Copy .caffemodel to ~/model/mine/
C. run python test.py


