import tensorflow as tf

def build_model():
    model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)])
    return model

def compile_model(model):
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    return model

def train_model(model,train_images,train_labels):
    model.fit(train_images, train_labels, epochs=10)

def evaluate_model(model, test_images,  test_labels):
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    return test_loss, test_acc

def test_model(model, X):
    probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
    Y = probability_model.predict(X)
    return Y