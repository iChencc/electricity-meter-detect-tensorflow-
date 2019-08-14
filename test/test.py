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
accuracy = tf.reduce_mean(tf.cast(correct_predition, tf.float32))                            #正确率预测

saver = tf.train.Saver()

sess = tf.InteractiveSession()
saver.restore(sess, 'save/')        #提取以前训练的参数值


images_test = np.empty((100, 200, 3), dtype='float32')  #声明用来测试图片的变量
images_cut= np.empty((244, 40, 40, 3), dtype='float32')      #声明裁剪图片的变量
images_show = np.empty((540, 960, 3), dtype='uint8')    #声明用来储存显示图片的变量
text = ['监测器', '开关', '温控器','油位计','油位计','油位计', '其他']
text1 = ['monitor', 'switch', 'temperature_controller','oil_level_gauge','oil_level_gauge','oil_level_gauge', 'other']

#打开测试文件夹得到图片
img_dir = file_name('test_photo/')
i = 0
for flie in img_dir:
    img = cv2.imread(flie)
    images_show = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    images_test = cv2.resize(img, None, fx=0.10416, fy=0.0925925, interpolation=cv2.INTER_CUBIC)
    images_test = images_test / 255.0
    cut_index = 0
    for n in range(4):
        for j in range(17 - n*2):
            for k in range(7 - n*2):
                images_cut[cut_index]  = cv2.resize(images_test[k*10 : k*10+40+20*n, j*10 : j*10+40+20*n], None, fx=40 / (40+20*n), fy=40 / (40+20*n), interpolation=cv2.INTER_CUBIC)
                cut_index = cut_index + 1
    y_val = y_conv.eval(feed_dict={x: images_cut, keep_prob: 1.0})
    for l in range(cut_index):
        y_val[l][6] = 0
    position_index = np.argmax(y_val)
    if (y_val[position_index // 7][position_index % 7] > 0.98):
        print(text[position_index % 7])
        n_1, j_1, k_1 = return_position(position_index)
        images = cv2.rectangle(images_show, (int(j_1*10*4.8), int(k_1*10*5.4)),
                               (int((j_1*10+40+20*n_1)*4.8), int((k_1*10+40+20*n_1)*5.4)),
                               (10, 255, 100), 2)
        cv2.putText(images, text1[position_index % 7], (int(j_1*10*4.8), int(k_1*10*5.4)+25),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (100, 255, 10))
        cv2.putText(images, str(y_val[position_index // 7][position_index % 7]),
                    (int(j_1*10*4.8), int(k_1*10*5.4)+50), cv2.FONT_HERSHEY_DUPLEX, 1, (100, 255, 10))
        cv2.imwrite("result_photo/" + text1[position_index % 7] + "__" + str(i) + ".jpg", images)
    else:
        print("没有找到"+str(y_val[position_index // 7][position_index % 7]))
        images = images_show
        cv2.putText(images, "Not Find Anything", (400, 250), cv2.FONT_HERSHEY_DUPLEX, 1, (100, 255, 10))
        cv2.imwrite("result_photo/" + "Not Find Anything" + "__" + str(i) + ".jpg", images)
    i = i + 1
    del y_val



