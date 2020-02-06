# yolo3
文件组织架构
checkpoints   训练后的模型
config  配置文件
data  数据集
logs  日志
output   输出预测（运行detect.py后生成的图片）
utils    使用的函数    
weights 权重模型
models.py #模型
train.py  训练
test.py   测试
detect.py 快速使用模型
requirements.txt  环境

相关准备
https://github.com/zyl067459/yolo3
首先从上述链接上将pytorch框架clone下来，放在pycharm的工程目录下

数据装载
将数据集Annotations、JPEGImages复制到YOLOV3工程目录下的data文件下；同时新建两个文件夹，分别命名为ImageSets和labels，最后我们将JPEGImages文件夹复制粘贴一下，并将文件夹重命名为images
分别运行makeTxt.py和voc_label.py
运行makeTxt.py后ImagesSets后面会出现四个文件
运行voc_label.py后labels下出现相应txt文件
注意这里的txt文件会出现空的文件，把这些空的文件删除，并且把train.txt中相应的文件删除，不然会报错

配置文件
在config中有coco.data,和custom.data，coco.data用的是coco2014的数据集，而custom.data用的是自定义数据集
这里我们使用自定义数据集，配置内容如下

classes= 2
train=data/train.txt
valid=data/val.txt
names=data/classes.names
backup=backup/
eval=coco

再在data目录下创建classes.names内容如下（github上names文件在data/custom下，注意修改）

hat
person

训练
train.py

 parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/custom.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    opt = parser.parse_args()
    print(opt)

epochs可以自己定义，不一定要10
如果出现内存溢出的情况，可以将batch_size调小
要训​​练自定义数据集，请运行：
$ python train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data
上面代码已经写好model_def和data_config，可以直接运行train.py

训练记录
---- [Epoch 7/100, Batch 7300/14658] ----
+------------+--------------+--------------+--------------+
| Metrics    | YOLO Layer 0 | YOLO Layer 1 | YOLO Layer 2 |
+------------+--------------+--------------+--------------+
| grid_size  | 16           | 32           | 64           |
| loss       | 1.554926     | 1.446884     | 1.427585     |
| x          | 0.028157     | 0.044483     | 0.051159     |
| y          | 0.040524     | 0.035687     | 0.046307     |
| w          | 0.078980     | 0.066310     | 0.027984     |
| h          | 0.133414     | 0.094540     | 0.037121     |
| conf       | 1.234448     | 1.165665     | 1.223495     |
| cls        | 0.039402     | 0.040198     | 0.041520     |
| cls_acc    | 44.44%       | 43.59%       | 32.50%       |
| recall50   | 0.361111     | 0.384615     | 0.300000     |
| recall75   | 0.222222     | 0.282051     | 0.300000     |
| precision  | 0.520000     | 0.300000     | 0.070175     |
| conf_obj   | 0.599058     | 0.622685     | 0.651472     |
| conf_noobj | 0.003778     | 0.004039     | 0.004044     |
+------------+--------------+--------------+--------------+
Total Loss 4.429395
---- ETA 0:35:48.821929


训练后会在checkpoints下出现生成的模型

测试
$ python test.py --weights_path checkpoints/yolov3_ckpt_0.pth
checkpoints/yolov3_ckpt_0.pth为训练后生成的模型，如果test中已经定义好weight_path，则只需python test.py

Average Precisions:
+ Class '0' (hat) - AP: 0.11766899769330798
+ Class '1' (person) - AP: 0.01315014824846156
mAP: 0.06540957297088477

快速使用模型
python detect.py --image_folder data/samples/
