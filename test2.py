import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import model_selection

tf.logging.info('TensorFlow')
tf.logging.set_verbosity(tf.logging.ERROR)
tf.logging.info('TensorFlow')

df=pd.read_csv('iris.csv')
df=pd.get_dummies(df,columns=['label'])
val=list(df.columns.values)
X=df.ix[:,(0,1,2,3)].values
Y=df.ix[:,(4,5,6)].values
print(X)
print(Y)

x_train,x_test,y_train,y_test=model_selection.train_test_split(X,Y,test_size=0.25,random_state=9999)

path=('C:\\Users\\Raghuram\\Desktop\\Python\\github\\iris\\saver')
n_hidden_1=30
n_hidden_2=30
n_input=x_train.shape[1]
n_classes=y_train.shape[1]
print(x_train.shape[0],x_train.shape[1])
weights={
    'h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
    'h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_1])),
    'out':tf.Variable(tf.random_normal([n_hidden_2,n_classes]))
    }
biases={
    'b1':tf.Variable(tf.random_normal([n_hidden_1])),
    'b2':tf.Variable(tf.random_normal([n_hidden_2])),
    'out':tf.Variable(tf.random_normal([n_classes]))
    }
training_epochs=1000
display_step=25

x=tf.placeholder('float',[None,n_input])
y=tf.placeholder('float',[None,n_classes])
keep_prob=tf.placeholder('float')

def neuralnet(x,weights,biases,keep_prob):
    layer_1=tf.add(tf.matmul(x,weights['h1']),biases['b1'])
    layer_1=tf.nn.relu(layer_1)
    layer_1=tf.nn.dropout(layer_1,rate=1-keep_prob)

    layer_2=tf.add(tf.matmul(layer_1,weights['h2']),biases['b2'])
    layer_2=tf.nn.relu(layer_2)
    layer_2=tf.nn.dropout(layer_2,rate=1-keep_prob)

    out_layer=tf.add(tf.matmul(layer_2,weights['out']),biases['out'])

    return out_layer

pred=neuralnet(x,weights,biases,keep_prob)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred,labels=y))
optimizer=tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
saver=tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,path)
    for epoch in range(training_epochs):
        avg_cost=0.
        _,c=sess.run([optimizer,cost],feed_dict={x:x_train,y:y_train,keep_prob:1.0})
        avg_cost+=c
        if(epoch%display_step==0):
            print("Epoch:",'%04d'%(epoch+1),"cost=","{:.9f}".format(avg_cost))
    print("Optimization Finished!")
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: x_test, y: y_test,keep_prob:1.0}))
    sp=saver.save(sess,save_path=path)
    print('Madel saved at:',sp)
    writer=tf.summary.FileWriter('C:\\Users\\Raghuram\\Desktop\\Python\\github\\iris\\graph',sess.graph)
    sess.close()
