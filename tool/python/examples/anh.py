#!/usr/bin/env python
#/************************************************************
#*
#* Licensed to the Apache Software Foundation (ASF) under one
#* or more contributor license agreements.  See the NOTICE file
#* distributed with this work for additional information
#* regarding copyright ownership.  The ASF licenses this file
#* to you under the Apache License, Version 2.0 (the
#* "License"); you may not use this file except in compliance
#* with the License.  You may obtain a copy of the License at
#*
#*   http://www.apache.org/licenses/LICENSE-2.0
#*
#* Unless required by applicable law or agreed to in writing,
#* software distributed under the License is distributed on an
#* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#* KIND, either express or implied.  See the License for the
#* specific language governing permissions and limitations
#* under the License.
#*
#*************************************************************/


import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from singa.model import *
from examples.datasets import anh_data

X_train, X_test, workspace = anh_data.load_data()

m = Sequential('cifar10-cnn', sys.argv)

# nb_filters, kernal, stride, pad
m.add(Convolution2D(10, (11,1), (1,1), (5,0), w_std=0.0001, b_lr=2))
m.add(MaxPooling2D(pool_size=(1), stride=2))
m.add(Activation('relu'))
m.add(LRN2D(1, alpha=0.00005, beta=0.75))



m.add(Convolution2D(10, (11,1), (1,1), (5,0), b_lr=2))
m.add(Activation('relu'))
m.add(AvgPooling2D(pool_size=(1), stride=2))
m.add(LRN2D(1, alpha=0.00005, beta=0.75))

'''
m.add(Convolution2D(10, (11,1), (1,1), (5,0)))
m.add(Activation('relu'))
m.add(AvgPooling2D(pool_size=(1), stride=2))
'''

m.add(Dense(2, w_wd=250, b_lr=2, b_wd=0, activation='softmax'))

sgd = SGD(decay=0.004, momentum=0.9, lr_type='manual', step=(0,60000,65000), step_lr=(0.001,0.0001,0.00001))
topo = Cluster(workspace)
m.compile(loss='categorical_crossentropy', optimizer=sgd, cluster=topo)

m.fit(X_train, nb_epoch=2000, with_test=True)
result = m.evaluate(X_test, test_steps=100, test_freq=125)

