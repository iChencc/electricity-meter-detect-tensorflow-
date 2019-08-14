import tensorflow as tf
import numpy as np
import cv2
import os    #加入其他模块

def get_weigth(shape):     #w权值
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)

def get_bias(shape):      #b权值
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, w):         #卷积核
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):     #池化层
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def file_name(file_dir):   #打开文件中的图片
    L=[]
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg' or '.png':
                L.append(os.path.join(root, file))
    return L

def dense_to_one_hot(labels_dense, num_classes):     #转化成one_hot编码
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def return_position(position):                    #返回最大权值的n,j,k坐标
    index = 0
    for n in range(4):
        for j in range(17 - n*2):
            for k in range(7 - n*2):
                if (index == position//7):
                    return n,j,k
                index  = index + 1

def get_next_data(images, labels):             #得到下一个batch训练数据
    global images_index
    position = images_index[1][0]
    images_batch = [images[position]] + [tf.image.flip_left_right(images[position]).eval()] + [tf.image.flip_up_down(images[position]).eval()]     #将图片上下和左右反转
    label_batch = [labels[position]] + [labels[position]] + [labels[position]]
    count = images_index[0][0]
    for i in range(1, 7):
        position = images_index[1][i] + count
        images_batch  = images_batch + [images[position]] + [tf.image.flip_left_right(images[position]).eval()] + [tf.image.flip_up_down(images[position]).eval()]
        label_batch = label_batch + [labels[position]] + [labels[position]] + [labels[position]]
        count = count + images_index[0][i]
    for j in range(7):
        images_index[1][j] = images_index[1][j] + 1
        if (images_index[1][j] >= images_index[0][j]):
            images_index[1][j] = 0
    return images_batch, label_batch

images_index = np.zeros((2,7), 'int32')
img_dir = file_name('train_data/monitor/')
for flie in img_dir:
    images_index[0][0] = images_index[0][0]+1
img_dir = file_name('train_data/switch/')
for flie in img_dir:
    images_index[0][1] = images_index[0][1] + 1
img_dir = file_name('train_data/temperature_controller/')
for flie in img_dir:
    images_index[0][2] = images_index[0][2] + 1
img_dir = file_name('train_data/oil_level_gauge_1/')
for flie in img_dir:
    images_index[0][3] = images_index[0][3] + 1
img_dir = file_name('train_data/oil_level_gauge_2/')
for flie in img_dir:
    images_index[0][4] = images_index[0][4] + 1
img_dir = file_name('train_data/oil_level_gauge_3/')
for flie in img_dir:
    images_index[0][5] = images_index[0][5] + 1
img_dir = file_name('train_data/other/')
for flie in img_dir:
    images_index[0][6] = images_index[0][6] + 1
images_count = 0
for i in range(7):
    images_count = images_count + images_index[0][i]     #打开训练文件夹  得到文件夹下面一共有多少张图片
print("一共"+str(images_count)+"张训练图片")                                      #显示文件总数


images = np.empty((int(images_count), 40, 40, 3), 'float32')    #申明用来存储图片的数组
labels = np.empty((int(images_count)), 'int32')                 #声明用来储存标签的数组
index = 0
#打开每个文件夹得到里面的图片
img_dir = file_name('train_data/monitor/')
for flie in img_dir:
    img = cv2.imread(flie)
    images[index] = cv2.resize(img, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_CUBIC)
    labels[index] = 0
    index = index + 1
img_dir = file_name('train_data/switch/')
for flie in img_dir:
    img = cv2.imread(flie)
    images[index] = cv2.resize(img, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_CUBIC)
    labels[index] = 1
    index = index + 1
img_dir = file_name('train_data/temperature_controller/')
for flie in img_dir:
    img = cv2.imread(flie)
    images[index] = cv2.resize(img, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_CUBIC)
    labels[index] = 2
    index = index + 1
img_dir = file_name('train_data/oil_level_gauge_1/')
for flie in img_dir:
    img = cv2.imread(flie)
    images[index] = cv2.resize(img, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_CUBIC)
    labels[index] = 3
    index = index + 1
img_dir = file_name('train_data/oil_level_gauge_2/')
for flie in img_dir:
    img = cv2.imread(flie)
    images[index] = cv2.resize(img, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_CUBIC)
    labels[index] = 4
    index = index + 1
img_dir = file_name('train_data/oil_level_gauge_3/')
for flie in img_dir:
    img = cv2.imread(flie)
    images[index] = cv2.resize(img, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_CUBIC)
    labels[index] = 5
    index = index + 1
img_dir = file_name('train_data/other/')
for flie in img_dir:
    img = cv2.imread(flie)
    images[index] = cv2.resize(img, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_CUBIC)
    labels[index] = 6
    index = index + 1
images = images / 255.0                     #将像素值转化成0~1之间的数据
labels = dense_to_one_hot(labels, 7)        #将标签转化为one_hot编码

print("标签和图片初始化完成")


x = tf.placeholder(tf.float32, [None, 40, 40, 3])    #图片数据输入节点
y_ = tf.placeholder(tf.float32, [None, 7])           #标签数据输入节点
keep_prob = tf.placeholder(tf.float32)

#第一个卷积层和池化层
w_conv1 = get_weigth([3, 3, 3, 128])
b_conv1 = get_bias([128])
h_conv1 = tf.nn.relu(conv2d(x, w_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#第二个卷积层和池化层
w_conv2 = get_weigth([3, 3, 128, 16])
b_conv2 = get_bias([16])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#第一个全连接层
w_fc1 = get_weigth([10*10*16, 512])
b_fc1 = get_bias([512])
h_pool2_flat = tf.reshape(h_pool2, [-1, 10*10*16])
h_fc1  = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1)+b_fc1)

#第二个全连接层和输出层
h_pool2_flat_drop = tf.nn.dropout(h_fc1, keep_prob)
w_fc2 = get_weigth([512, 7])
b_fc2 = get_bias([7])
y_conv = tf.nn.softmax(tf.matmul(h_pool2_flat_drop, w_fc2)+b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv), reduction_indices=[1]))     #损失函数
train_step = tf.train.AdamOptimizer(0.00001).minimize(cross_entropy)                         #Adam 优化器

correct_predition = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predition, tf.float32))                            #正确率预测            #正确率预测


saver = tf.train.Saver()
sess = tf.InteractiveSession()
saver.restore(sess, 'save/')        #提取以前训练的参数值
print("测试全部图片的正确率为"+str(accuracy.eval(feed_dict={x: images[0:2000], y_: labels[0:2000], keep_prob: 1.0})))

for i in range(20000):
    batch_xs, batch_ys = get_next_data(images, labels)    #得到下一次的值
    train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.75})    #开始训练
    print(i)
    if (i+1) % 20 == 0:
        saver.save(sess, 'save/')           #每训练20次就储存一次训练的数据
    if(i+1)%100 == 0:
        #print("训练了"+str(i+1)+"次后的正确结果为："+str(accuracy.eval(feed_dict={x: images, y_: labels, keep_prob: 1.0})))
        print("训练了"+str(i+1)+"次后的正确结果为："+str(accuracy.eval(feed_dict={x: images[0:500], y_: labels[0:500], keep_prob: 1.0})))

sess.close()
