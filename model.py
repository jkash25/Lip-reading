
model = Sequential()
model.add(Conv3D(32, (3, 3, 3), strides = 1, input_shape=(10, 100, 100, 1), activation='relu', padding='valid'))
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2))
model.add(Conv3D(64, (3, 3, 3), activation='relu', strides=1))
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=2))
shape = (1,10,10,128)
model.add((Flatten()))
model.add(Dense(10, activation='softmax'))
model.compile(loss = 'categorical_crossentropy',optimizer = 'Adam',metrics = ['accuracy'])
model.summary()
